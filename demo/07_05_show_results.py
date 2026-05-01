#!/usr/bin/env python3

"""
Step 5: Show and compare results from brute-force and NAPP evaluations.

Run from the demo directory after both evaluations complete:
    cd demo
    python 07_05_show_results.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path('/tempory/the_three_potatoes/ri_project/workspaces/amelie/FlexNeuART-IR-TTP')
COLLECT_ROOT = (REPO_ROOT / 'demo' / 'collections').resolve()
COLLECT_NAME = 'stackoverflow_all'
TEST_PART = 'test'
TRAIN_PART = 'train'
TRAIN_CAND_QTY = 1000

collect_dir = COLLECT_ROOT / COLLECT_NAME

def parse_rep_file(rep_file: Path):
    metrics = {}
    with rep_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, value = line.split(':', 1)
            value = value.strip().split()[0]
            try:
                metrics[key.strip()] = float(value)
            except ValueError:
                continue
    return metrics

def collect_rep_dir(rep_dir: Path):
    rows = []
    for rep_file in sorted(rep_dir.glob('out_*.rep'), key=lambda p: int(p.stem.split('_')[1])):
        top_k = int(rep_file.stem.split('_')[1])
        row = {'top_k': top_k}
        row.update(parse_rep_file(rep_file))
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f'No rep files found in {rep_dir}')
    return rows

print('=' * 80)
print('SHOW AND COMPARE RESULTS')
print('=' * 80)
print()

bruteforce_rep_dir = collect_dir / 'results' / TEST_PART / 'feat_exper' / 'bm25_model1_optimal_bruteforce' / 'rep'
napp_rep_dir = collect_dir / 'results' / TEST_PART / 'feat_exper' / 'bm25_model1_optimal_napp' / 'rep'

if not bruteforce_rep_dir.exists():
    print(f'ERROR: Brute-force report directory not found: {bruteforce_rep_dir}')
    sys.exit(1)

if not napp_rep_dir.exists():
    print(f'ERROR: NAPP report directory not found: {napp_rep_dir}')
    sys.exit(1)

def to_rows(rep_dir: Path, label: str):
    rows = collect_rep_dir(rep_dir)
    for row in rows:
        row['system'] = label
    return rows

bruteforce_rows = to_rows(bruteforce_rep_dir, 'brute_force')
napp_rows = to_rows(napp_rep_dir, 'napp')

metrics_to_show = ['MAP', 'NDCG@10', 'NDCG@20', 'NDCG@100', 'P@20', 'MRR', 'RECALL']
all_top_ks = sorted({row['top_k'] for row in bruteforce_rows} | {row['top_k'] for row in napp_rows})

def metric_value(rows, top_k, metric):
    for row in rows:
        if row['top_k'] == top_k:
            return row.get(metric, float('nan'))
    return float('nan')

header = ['top_k'] + [f'{metric}_bruteforce' for metric in metrics_to_show] + [f'{metric}_napp' for metric in metrics_to_show]
print('\t'.join(header))
for top_k in all_top_ks:
    values = [str(top_k)]
    for metric in metrics_to_show:
        values.append(f'{metric_value(bruteforce_rows, top_k, metric):.6f}')
    for metric in metrics_to_show:
        values.append(f'{metric_value(napp_rows, top_k, metric):.6f}')
    print('\t'.join(values))

def best_map(rows):
    if not rows:
        return None
    return max(rows, key=lambda row: row.get('MAP', float('-inf')))

best_bruteforce = best_map(bruteforce_rows)
best_napp = best_map(napp_rows)
print()
print('Best brute-force MAP row:', best_bruteforce)
print('Best NAPP MAP row       :', best_napp)
print()

print('=' * 80)
print('COMPARISON COMPLETE')
print('=' * 80)
