#!/usr/bin/env python3
"""Frozen development selection followed by ten matched evaluation orders."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def score(efficacy, locality):
    return 2 * efficacy * locality / (efficacy + locality) if efficacy + locality else 0


def validate(task, output):
    path = output / task['id'] / 'results.json'
    record = json.loads(path.read_text())
    assert record['status'] == 'complete'
    assert record['n_edits'] == task['n_edits']
    assert record['lora']['rank'] == task['rank']
    assert record['ssr']['lambda'] == task['coefficient']
    assert record['dual_endpoint_protocol']['rehearsal_window'] == task['window']
    assert record['stream_seed'] == task['seed']
    for checkpoint in record['checkpoints']:
        history = checkpoint['history']
        n = checkpoint['after_edits']
        assert history['n_evaluated'] == history['n_history'] == n
        assert [r['history_position'] for r in history['evaluations']] == list(range(n))
        for metric, item_metric in [('efficacy', 'retained_efficacy'),
                                    ('locality_target_consistency', 'retained_locality_target_consistency')]:
            values = [r[item_metric] for r in history['evaluations']]
            assert all(v is not None and math.isfinite(v) for v in values)
            assert math.isclose(history[metric], 100 * sum(values)/n, abs_tol=1e-8)
        assert checkpoint.get('adapter_checkpoint'), 'Missing saved adapter'
    answers = Path(record['generated_answers']['path'])
    assert hashlib.sha256(answers.read_bytes()).hexdigest() == record['generated_answers']['sha256']
    log = (output / 'logs' / (task['id'] + '.log')).read_text()
    losses = [float(line.split()[-1]) for line in log.splitlines() if line.startswith('Batch loss ')]
    assert len(losses) == task['n_edits'] * task['steps'] and all(math.isfinite(v) for v in losses)
    return {'task': task, 'history_efficacy': record['final_history_efficacy'],
            'history_locality': record['final_history_locality'],
            'immediate_efficacy': 100 * record['immediate']['efficacy'],
            'immediate_locality': 100 * record['immediate']['locality_target_consistency'],
            'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def run_one(task, gpu, output, config):
    stem = output / task['id']
    stem.mkdir(parents=True, exist_ok=True)
    path = stem / 'results.json'
    assert not path.exists(), path
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu),
           'SSR_REHEARSAL_WINDOW': str(task['window']), 'SSR_RESULT_PATH': str(path),
           'SSR_REPO_DIR': str(PROJECT), 'EASYEDIT_DIR': config['easyedit'],
           'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
           'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1'}
    command = [sys.executable, '-u', str(PROJECT / 'scripts/run_fig5_dual_endpoints.py'),
        '--method', 'LoRA', '--model-path', config['model'], '--model-label', 'Qwen2.5-7B-Instruct',
        '--hparams-root', str(Path(config['easyedit']) / 'hparams'), '--hparams-stem', 'qwen2.5-7b',
        '--dataset', task['dataset'], '--stream-manifest', task['stream'],
        '--n-edits', str(task['n_edits']), '--lora-rank', str(task['rank']), '--lora-lr', '0.0005',
        '--lora-steps', str(task['steps']), '--seed', str(task['seed']),
        '--ssr-lambda', str(task['coefficient']), '--history-checkpoints', *map(str, task['checkpoints']),
        '--history-max-samples', str(task['n_edits']), '--pre-edit-max-samples', '16',
        '--method-cache-dir', str(stem / 'cache'), '--checkpoint-dir', str(stem / 'adapters'),
        '--output', str(path)]
    print(json.dumps({'event': 'start', 'gpu': gpu, 'task': task['id']}), flush=True)
    with (output / 'logs' / (task['id'] + '.log')).open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    row = validate(task, output)
    print(json.dumps({'event': 'complete', 'gpu': gpu, **row}), flush=True)
    return row


def run_stage(tasks, output, config):
    def worker(gpu):
        return [run_one(task, gpu, output, config) for task in tasks[gpu::8]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        batches = list(executor.map(worker, range(8)))
    return [row for batch in batches for row in batch]


def select(rows, rank):
    def group(window, coefficient):
        found = [r for r in rows if r['task']['rank'] == rank and
                 r['task']['window'] == window and r['task']['coefficient'] == coefficient]
        assert len(found) == 2
        e = sum(r['history_efficacy'] for r in found)/2
        l = sum(r['history_locality'] for r in found)/2
        return score(e, l)
    # Shared rehearsal window is chosen using only the plain baseline.
    window = max((8, 32), key=lambda w: (group(w, 0), -w))
    coefficient = max((0.01, 0.03), key=lambda c: (group(window, c), -c))
    return {'rank': rank, 'window': window, 'coefficient': coefficient,
            'rule': 'window: plain history E/L harmonic mean; coefficient: SSR same score; ties smaller value',
            'plain_development_score': group(window, 0), 'ssr_development_score': group(window, coefficient)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=PROJECT, text=True).strip() == ''
    assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=config['easyedit'], text=True).strip() == ''
    count = sum(line.startswith('GPU ') for line in subprocess.check_output(['nvidia-smi', '-L'], text=True).splitlines())
    assert count == 8, count
    for row in config['inputs']:
        assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256']
    output = args.output.resolve()
    assert not output.exists(), output
    (output / 'logs').mkdir(parents=True)
    write(output / 'config.json', config)
    # Both arms, both ranks, both windows exercise a real forward/backward.
    canaries = []
    for rank in (8, 32):
        for window in (8, 32):
            for coefficient in (0, 0.03):
                task = dict(config['development'][0], rank=rank, window=window,
                            coefficient=coefficient, n_edits=2, steps=2, checkpoints=[1, 2])
                task['id'] = f'canary_r{rank}_w{window}_c{coefficient}'
                canaries.append(task)
    run_stage(canaries, output, config)
    write(output / 'canary.complete.json', {'n': len(canaries), 'status': 'complete'})
    rows = run_stage(config['development'], output, config)
    write(output / 'development.complete.json', rows)
    selections = [select(rows, rank) for rank in (8, 32)]
    write(output / 'selection.json', {'selected_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                     'selections': selections})
    tasks = []
    for selection in selections:
        for stream in config['evaluation_streams']:
            for coefficient in (0, selection['coefficient']):
                rank, seed = selection['rank'], stream['seed']
                tasks.append({**stream, 'id': f'eval_r{rank}_s{seed}_c{coefficient}',
                              'rank': rank, 'window': selection['window'], 'coefficient': coefficient,
                              'n_edits': 100, 'steps': 20, 'checkpoints': [1, 2, 5, 10, 25, 50, 100]})
    write(output / 'evaluation_plan.json', tasks)
    results = run_stage(tasks, output, config)
    assert len(results) == 40
    write(output / 'evaluation.complete.json', results)


if __name__ == '__main__':
    main()
