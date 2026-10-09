"""Verify the current manuscript's source data and release inventory."""

import ast
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'Data_S1'
checks = 0


def check(condition, message):
    global checks
    if not condition:
        raise AssertionError(message)
    checks += 1


def rows(path):
    with (DATA / path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def close(actual, expected, label):
    check(math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-8), label)


def paired(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def verify_inventory():
    manifest = rows('MANIFEST.csv')
    names = {row['file'] for row in manifest}
    actual = {str(file.relative_to(DATA)) for file in DATA.rglob('*')
              if file.is_file() and file.name != 'MANIFEST.csv'}
    check(len(manifest) == len(names) == len(actual) and names == actual, 'Data S1 inventory')
    for row in manifest:
        check(digest(DATA / row['file']) == row['delivered_sha256'], 'Data S1 hash ' + row['file'])

    release = list(csv.DictReader((ROOT / 'FILE_MANIFEST.csv').open(newline='')))
    listed = {row['path'] for row in release}
    tracked = subprocess.check_output(
        ['git', 'ls-files', '-co', '--exclude-standard', '-z'], cwd=ROOT,
    ).decode().split('\0')
    actual = {name for name in tracked if name and name != 'FILE_MANIFEST.csv' and (ROOT / name).is_file()}
    check(len(release) == len(listed) and listed == actual, 'repository inventory')
    for row in release:
        file = ROOT / row['path']
        check(file.stat().st_size == int(row['bytes']) and digest(file) == row['sha256'],
              'release hash ' + row['path'])


def verify_figure4():
    displayed = rows('figure4/Fig4G_displayed_transfer.csv')
    expected = [(0, 5.5), (0, 23), (3, 8.18), (12, 16.5), (5, 1.64), (10, 3.36)]
    check(len(displayed) == 6, '4G six original full-recipe settings')
    for row, (efficacy, locality) in zip(displayed, expected):
        check(row['n'] == '1' and 'complete recipe' in row['attribution'], '4G recipe and n')
        close(row['efficacy_gain_pp'], efficacy, '4G original efficacy')
        close(row['locality_gain_pp'], locality, '4G original locality')
    displayed = rows('figure4/Fig4H_history_rerun/seed_level.csv')
    summary = rows('figure4/Fig4H_history_rerun/summary.csv')
    check(len(displayed) == 20 and len(summary) == 2, '4H rerun inventory')
    by = {(row['method'], int(row['seed']), row['arm']): row for row in displayed}
    for row in summary:
        method, metric = row['method'], row['metric']
        seeds = list(range(9941, 9951))
        left = np.array([float(by[method, seed, 'control'][metric]) for seed in seeds])
        right = np.array([float(by[method, seed, 'ssr'][metric]) for seed in seeds])
        check(int(row['n_pairs']) == 10, '4H complete pairs')
        close(left.mean(), row['control_percent'], '4H control')
        close(right.mean(), row['ssr_percent'], '4H SSR')
        close((right-left).mean(), row['gain_pp'], '4H final-history gain')
        for seed in seeds:
            for arm in ('control', 'ssr'):
                record = DATA / 'figure4/Fig4H_history_rerun' / by[method, seed, arm]['result']
                check(record.is_file() and digest(record) == by[method, seed, arm]['result_sha256'],
                      '4H raw result hash')
                raw = json.loads(record.read_text())['checkpoints'][-1]['history']
                check(raw['n_evaluated'] == raw['n_locality_evaluated'] == 250,
                      '4H both final-history endpoints')


def verify_figure5():
    source = rows('figure5_rehearsal/seed_level.csv')
    summary = rows('figure5_rehearsal/complete_rank_summary.csv')
    check(len(source) == 60 and len(summary) == 9, '5 main complete records')
    by = {(int(row['rank']), int(row['seed']), row['arm']): row for row in source}
    for row in source:
        record = DATA / 'figure5_rehearsal/records' / row['result']
        check(record.is_file(), '5 main raw result exists')
        check(digest(record) == row['result_sha256'], '5 main raw result hash')
    for row in summary:
        rank, metric = int(row['rank']), row['metric']
        seeds = sorted({seed for r, seed, arm in by if r == rank and arm == 'plain'})
        check(len(seeds) == int(row['n_pairs']) == 10, '5 main ten pairs')
        left = [float(by[rank, seed, 'plain'][metric]) for seed in seeds]
        right = [float(by[rank, seed, 'ssr'][metric]) for seed in seeds]
        mean, low, high = paired(np.asarray(right) - left)
        for actual, key in [(np.mean(left), 'plain_mean'), (np.mean(right), 'ssr_mean'),
                            (mean, 'delta_pp'), (low, 'ci95_low_pp'), (high, 'ci95_high_pp')]:
            close(actual, row[key], '5 main ' + key)

    source = rows('figure5_dual_followup/seed_level.csv')
    summary = rows('figure5_dual_followup/summary.csv')
    check(len(source) == 40 and len(summary) == 8, '5 SI complete records')
    by = {(int(row['rank']), int(row['seed']), row['arm']): row for row in source}
    for row in summary:
        rank, metric = int(row['rank']), row['metric']
        field = {'history_efficacy': 'history_efficacy', 'history_locality': 'history_locality',
                 'immediate_efficacy': 'immediate_efficacy', 'immediate_locality': 'immediate_locality'}[metric]
        seeds = sorted({seed for r, seed, arm in by if r == rank and arm == 'plain'})
        check(len(seeds) == int(row['n_pairs']) == 10, '5 SI pair count')
        if rank == 32:
            check(26092721 in seeds and 26092720 not in seeds, '5 rank32 displayed orders')
        else:
            check(26092720 in seeds and 26092721 not in seeds, '5 rank8 displayed orders')
        left = [float(by[rank, seed, 'plain'][field]) for seed in seeds]
        right = [float(by[rank, seed, 'ssr'][field]) for seed in seeds]
        mean, low, high = paired(np.asarray(right) - left)
        for actual, key in [(np.mean(left), 'plain_mean'), (np.mean(right), 'ssr_mean'),
                            (mean, 'delta_pp'), (low, 'low'), (high, 'high')]:
            close(actual, row[key], '5 SI ' + key)
        for seed in seeds:
            for arm in ('plain', 'ssr'):
                record = DATA / 'figure5_dual_followup/records' / (
                    f'eval_r{rank}_s{seed}_c' + ('0' if arm == 'plain' else by[rank, seed, arm]['coefficient'])
                ) / 'results.json'
                check(record.is_file(), '5 SI raw result exists')
                check(digest(record) == by[rank, seed, arm]['result_sha256'], '5 SI raw result hash')

    source = rows('figure5_rank1664_followup/seed_level.csv')
    summary = rows('figure5_rank1664_followup/summary.csv')
    check(len(source) == 20 and len(summary) == 4, '5 extended complete records')
    by = {(int(row['rank']), int(row['seed']), row['arm']): row for row in source}
    check(len(by) == 20, '5 extended distinct run identities')
    for row in source:
        rank, seed = int(row['rank']), int(row['seed'])
        check(rank == 16 and seed in range(26092811, 26092821), '5 extended fixed cohort')
        record = DATA / 'figure5_rank1664_followup' / row['result']
        check(record.is_file() and digest(record) == row['result_sha256'], '5 extended raw result hash')
        check((record.parent / 'generated_answers.jsonl').is_file(), '5 extended raw generations')
    for row in summary:
        rank, metric = int(row['rank']), row['metric']
        seeds = list(range(26092811, 26092821))
        check(int(row['n_pairs']) == 10, '5 extended ten pairs')
        left = [float(by[rank, seed, 'plain'][metric]) for seed in seeds]
        right = [float(by[rank, seed, 'ssr'][metric]) for seed in seeds]
        mean, low, high = paired(np.asarray(right) - left)
        for actual, key in [(np.mean(left), 'plain_mean'), (np.mean(right), 'ssr_mean'),
                            (mean, 'delta_pp'), (low, 'ci95_low_pp'), (high, 'ci95_high_pp')]:
            close(actual, row[key], '5 extended ' + key)


def verify_identity():
    identity = json.loads((ROOT / 'MANUSCRIPT_IDENTITY.json').read_text())
    for name in ('Figure4.pdf', 'Figure5.pdf', 'Figure6.pdf'):
        check(digest(ROOT / 'figures/reference' / name) == identity['sha256']['figures/' + name],
              'manuscript reference ' + name)
    for file in ROOT.rglob('*.py'):
        if '.git' not in file.parts and 'build' not in file.parts:
            ast.parse(file.read_text(), filename=str(file))
            check(True, 'Python syntax ' + str(file))

    strengths = rows('figure6/Fig6F_supplementary_nine_strengths.csv')
    check(len(strengths) == 50, 'nine-strength supplementary source size')
    settings = {row['candidate_id'] for row in strengths if row['method'] == 'kd_ssr'}
    check(len(settings) == 9 and all(
        sum(row['candidate_id'] == setting and row['method'] == 'kd_ssr' for row in strengths) == 5
        for setting in settings), 'nine displayed strengths x five seeds')


def verify_figure5_focus():
    mapping = json.loads((DATA / 'figure5_panel_sources.json').read_text())
    focus = mapping['A,B,E']
    displayed = [26092711, 26092712, 26092713, 26092714, 26092715,
                 26092716, 26092717, 26092718, 26092719, 26092721]
    check(focus['rank'] == 32 and focus['seeds'] == displayed and 'replacement_seed' not in focus
          and focus.get('rehearsal_window') == 32,
          '5 focus complete cohort identity')
    def cohort(panel, ranks):
        return next(item for item in mapping[panel]['cohorts'] if item['ranks'] == ranks)
    check(cohort('C', [8])['directory'] == 'figure5_dual_followup'
          and cohort('C', [8])['metrics'] == ['immediate_efficacy', 'history_efficacy']
          and cohort('C', [32])['seeds'] == displayed
          and cohort('C', [16])['ranks'] == [16]
          and cohort('D', [8])['metrics'] == ['immediate_locality', 'history_locality']
          and cohort('D', [32])['seeds'] == displayed
          and cohort('D', [16])['ranks'] == [16],
          '5 C and D are 32-target absolute endpoints')
    source = {(int(row['seed']), row['arm']): row
              for row in rows('figure5_dual_followup/seed_level.csv') if row['rank'] == '32'}
    checkpoints = {}
    for (seed, arm), row in source.items():
        file = DATA / 'figure5_dual_followup/records' / f"eval_r32_s{seed}_c{row['coefficient']}" / 'results.json'
        record = json.loads(file.read_text())
        for cp in record['checkpoints']:
            checkpoints[seed, arm, int(cp['after_edits'])] = cp
    check(len(source) == 20 and len(checkpoints) == 140, '5 focus complete checkpoint set')
    for filename, geometry in [('trajectory_summary.csv', False), ('geometry_summary.csv', True)]:
        summary = rows('figure5_dual_followup/' + filename)
        check(len(summary) == (14 if geometry else 28), '5 focus summary dimensions ' + filename)
        for row in summary:
            edit, metric = int(row['after_edits']), row['metric']
            check(row['rank'] == '32' and row['n_pairs'] == '10', '5 focus rank and n')
            values = []
            for arm in ('plain', 'ssr'):
                arm_values = []
                for seed in focus['seeds']:
                    cp = checkpoints[seed, arm, edit]
                    if geometry:
                        value = cp['lora_geometry'][metric]
                    else:
                        history = metric.startswith('history_')
                        key = 'efficacy' if metric.endswith('_efficacy') else 'locality_target_consistency'
                        value = cp['history' if history else 'immediate'][key] * (1 if history else 100)
                    arm_values.append(value)
                values.append(np.asarray(arm_values))
            mean, low, high = paired(values[1] - values[0])
            for actual, key in [(values[0].mean(), 'plain_mean'), (values[1].mean(), 'ssr_mean'),
                                (mean, 'delta' if geometry else 'delta_pp'), (low, 'low'), (high, 'high')]:
                close(actual, row[key], '5 focus raw-to-summary ' + metric + ' ' + key)


if __name__ == '__main__':
    verify_inventory()
    verify_figure4()
    verify_figure5()
    verify_figure5_focus()
    verify_identity()
    print(json.dumps({'checks_passed': checks, 'gpu_training_rerun': False}))
