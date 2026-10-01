#!/usr/bin/env python3
"""Rehearsal with matched immediate/history evaluators and saved answers."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import run_lowrank_ke_rehearsal as rehearsal
from llm_ke.locality import iter_locality_items

runner = rehearsal.runner
OUTPUT = Path(os.environ['SSR_RESULT_PATH']).resolve()
ANSWERS = OUTPUT.with_name('generated_answers.jsonl')
CONTEXT = {'phase': 'pre_edit', 'after_edits': 0}
original_generate = runner.generate
original_evaluate_one = runner.evaluate_one
original_history = runner.evaluate_history


def generate(model, tokenizer, prompt):
    prediction = original_generate(model, tokenizer, prompt)
    with ANSWERS.open('a') as handle:
        handle.write(json.dumps({**CONTEXT, 'prompt': prompt, 'prediction': prediction}, ensure_ascii=False) + '\n')
    return prediction


def groups_with_boundary_match(groups, generate_fn):
    items = []
    for item in iter_locality_items(groups):
        prediction = generate_fn(item['prompt'])
        items.append({**item, 'prediction': prediction,
                      'matched': bool(rehearsal.boundary_target_match(prediction, item['ground_truths']))})
    return {'score': sum(row['matched'] for row in items) / max(1, len(items)),
            'n_items': len(items), 'evaluations': items,
            'evaluator': 'normalized_complete_phrase_boundary_v1'}


def evaluate_one(sample, model, tokenizer):
    CONTEXT.update(phase='immediate', after_edits=len(rehearsal.history))
    return original_evaluate_one(sample, model, tokenizer)


def evaluate_history(history, model, tokenizer, after_edits, max_samples):
    CONTEXT.update(phase='history', after_edits=after_edits)
    result = original_history(history, model, tokenizer, after_edits, max_samples)
    valid = []
    for item in result['evaluations']:
        sample = history[item['history_position']]
        scores = groups_with_boundary_match(sample.get('locality', {}),
                    lambda prompt: generate(model, tokenizer, prompt))
        value = scores['score'] if scores['n_items'] else None
        item.update(prompt=sample['prompt'], target_new=sample['target_new'],
                    retained_locality_target_consistency=value, locality=scores)
        if value is not None:
            valid.append(value)
    result.update(locality_target_consistency=100 * sum(valid) / len(valid) if valid else None,
                  locality_n=len(valid), locality_evaluator='normalized_complete_phrase_boundary_v1')
    CONTEXT['phase'] = 'pre_edit_reference_check'
    return result


runner.generate = generate
runner.evaluate_one = evaluate_one
runner.evaluate_history = evaluate_history
runner._evaluate_target_groups = groups_with_boundary_match


if __name__ == '__main__':
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if ANSWERS.exists():
        raise FileExistsError(ANSWERS)
    runner.main()
    record = json.loads(OUTPUT.read_text())
    project = Path(__file__).resolve().parents[1]
    record['dual_endpoint_protocol'] = {
        'protocol': 'fig5_rehearsal_dual_v1',
        'rehearsal_window': rehearsal.WINDOW,
        'rehearsal_selection': 'evenly_spaced_previous_targets_plus_current',
        'target_suffix_token_id': 151645,
        'efficacy_and_locality_match': 'normalized_complete_phrase_boundary_v1',
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project, text=True).strip(),
        'easyedit_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runner.EASYEDIT_ROOT, text=True).strip(),
        'source_sha256': {p.name: runner.sha256(p) for p in
            (Path(__file__), Path(rehearsal.__file__), Path(runner.__file__))},
    }
    record['final_history_locality'] = record['checkpoints'][-1]['history']['locality_target_consistency']
    record['generated_answers'] = {'path': str(ANSWERS), 'sha256': runner.sha256(ANSWERS)}
    runner.atomic_json(OUTPUT, record)
