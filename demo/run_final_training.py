#!/usr/bin/env python3
"""
Train Final BM25+Model1 Fusion on stackoverflow_all (Optimal Config)

This script takes the best parameters found during tuning, creates one final 
BM25+Model1 descriptor, trains on `train`, evaluates on `test`, and prints the report.

Run from the flexneuart_scripts directory:
    cd demo/flexneuart_scripts
    python ../run_final_training.py
"""

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# ============================================================================
# SETUP
# ============================================================================

REPO_ROOT = Path('/tempory/the_three_potatoes/ri_project/workspaces/amelie/FlexNeuART-IR-TTP')
SCRIPTS_DIR = REPO_ROOT / 'demo' / 'flexneuart_scripts'
COLLECT_ROOT = (REPO_ROOT / 'demo' / 'collections').resolve()
COLLECT_NAME = 'stackoverflow_all'

TRAIN_PART = 'train'
TEST_PART = 'test'  # evaluation on test set
TRAIN_CAND_QTY = 1000  # retrieve 1000 candidates for evaluation on 1-100

# top_k values used during tuning (differ between tuning runs)
BM25_TUNE_TOP_K = '250'    # BM25 tuned at top_k=250
MODEL1_TUNE_TOP_K = '15'   # Model1 tuned at top_k=15

# Final evaluation will use all N from 1-100 (from DEFAULT_TEST_CAND_QTY_LIST in config.sh)

# Metric used to pick best configuration from tuning tables
SELECT_METRIC_COL = 'MAP'

BM25_TUNE_TSV = SCRIPTS_DIR / 'bm25tune_stackoverflow_all.tsv'
MODEL1_TUNE_TSV = SCRIPTS_DIR / 'model1tune_stackoverflow_all.tsv'

# Output descriptor locations inside the collection
collect_dir = COLLECT_ROOT / COLLECT_NAME
desc_root = collect_dir / 'exper_desc.best'
extractor_dir = desc_root / 'extractors'
desc_root.mkdir(parents=True, exist_ok=True)
extractor_dir.mkdir(parents=True, exist_ok=True)

os.environ['COLLECT_ROOT'] = str(COLLECT_ROOT)

print('=' * 80)
print('SETUP')
print('=' * 80)
print('REPO_ROOT        =', REPO_ROOT)
print('SCRIPTS_DIR      =', SCRIPTS_DIR)
print('COLLECT_ROOT     =', COLLECT_ROOT)
print('COLLECT_NAME     =', COLLECT_NAME)
print('TRAIN_PART       =', TRAIN_PART)
print('TEST_PART        =', TEST_PART)
print('TRAIN_CAND_QTY   =', TRAIN_CAND_QTY)
print('BM25_TUNE_TOP_K  =', BM25_TUNE_TOP_K)
print('MODEL1_TUNE_TOP_K=', MODEL1_TUNE_TOP_K)
print('SELECT_METRIC_COL=', SELECT_METRIC_COL)
print('BM25_TUNE_TSV    =', BM25_TUNE_TSV)
print('MODEL1_TUNE_TSV  =', MODEL1_TUNE_TSV)
print()

# ============================================================================
# VALIDATE PREREQUISITES
# ============================================================================

print('=' * 80)
print('VALIDATE PREREQUISITES')
print('=' * 80)

required_paths = [
    BM25_TUNE_TSV,
    MODEL1_TUNE_TSV,
    collect_dir / 'input_data' / TRAIN_PART / 'QuestionFields.jsonl',
    collect_dir / 'input_data' / TRAIN_PART / 'AnswerFields.jsonl',
    collect_dir / 'input_data' / TRAIN_PART / 'qrels.txt',
    collect_dir / 'input_data' / TEST_PART / 'QuestionFields.jsonl',
    collect_dir / 'input_data' / TEST_PART / 'AnswerFields.jsonl',
    collect_dir / 'input_data' / TEST_PART / 'qrels.txt',
    collect_dir / 'lucene_index',
    collect_dir / 'forward_index' / 'text',
    collect_dir / 'derived_data' / 'giza' / 'text' / 'output.t1.5.bin'
]

for p in required_paths:
    if not p.exists():
        print(f'ERROR: Missing prerequisite: {p}')
        sys.exit(1)

print('All prerequisites found.')
print()

# ============================================================================
# PICK BEST PARAMS FROM TUNING TSVs
# ============================================================================

print('=' * 80)
print('PICK BEST PARAMS FROM TUNING TSVs')
print('=' * 80)

def load_best_row(tsv_path: Path, metric_col: str, top_k: str):
    with tsv_path.open('r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))

    if not rows:
        raise RuntimeError(f'No rows in TSV: {tsv_path}')

    filt = [r for r in rows if r.get('top_k') == str(top_k)]
    if not filt:
        raise RuntimeError(f'No rows with top_k={top_k} in {tsv_path}')

    missing = [r for r in filt if metric_col not in r or r[metric_col] in (None, '')]
    if missing:
        raise RuntimeError(f'Missing metric column {metric_col} in some rows of {tsv_path}')

    best = max(filt, key=lambda r: float(r[metric_col]))
    return best

def parse_bm25_from_exper_subdir(exper_subdir: str):
    m = re.search(r'k1=([0-9.]+)_b=([0-9.]+)', exper_subdir)
    if not m:
        raise RuntimeError(f'Cannot parse BM25 params from exper_subdir: {exper_subdir}')
    return float(m.group(1)), float(m.group(2))

def parse_model1_from_exper_subdir(exper_subdir: str):
    m_lambda = re.search(r'lambda=([0-9.]+)', exper_subdir)
    m_prob = re.search(r'probSelfTran=([0-9.]+)', exper_subdir)
    m_min = re.search(r'minTranProb=([0-9.eE+-]+)', exper_subdir)

    if not m_lambda or not m_prob:
        raise RuntimeError(f'Cannot parse Model1 params from exper_subdir: {exper_subdir}')

    lamb = float(m_lambda.group(1))
    prob_self = float(m_prob.group(1))
    min_prob = float(m_min.group(1)) if m_min else 2.5e-3

    return lamb, prob_self, min_prob

# Load best params from each TSV using their respective tuning top_k values
bm25_best = load_best_row(BM25_TUNE_TSV, SELECT_METRIC_COL, BM25_TUNE_TOP_K)
model1_best = load_best_row(MODEL1_TUNE_TSV, SELECT_METRIC_COL, MODEL1_TUNE_TOP_K)

BM25_K1, BM25_B = parse_bm25_from_exper_subdir(bm25_best['exper_subdir'])
MODEL1_LAMBDA, MODEL1_PROB_SELF_TRAN, MODEL1_MIN_PROB = parse_model1_from_exper_subdir(model1_best['exper_subdir'])

print('Best BM25 row (top_k=250):')
for k, v in bm25_best.items():
    print(f'  {k}: {v}')
print()
print('Best Model1 row (top_k=15):')
for k, v in model1_best.items():
    print(f'  {k}: {v}')
print()
print('Selected params:')
print('  BM25_K1 =', BM25_K1)
print('  BM25_B =', BM25_B)
print('  MODEL1_LAMBDA =', MODEL1_LAMBDA)
print('  MODEL1_PROB_SELF_TRAN =', MODEL1_PROB_SELF_TRAN)
print('  MODEL1_MIN_PROB =', MODEL1_MIN_PROB)
print()

# ============================================================================
# BUILD ONE FINAL DESCRIPTOR JSON (best config only)
# ============================================================================

print('=' * 80)
print('BUILD FINAL DESCRIPTOR')
print('=' * 80)

fid = (
    f'bm25=text+model1=text'
    f'+k1={BM25_K1:g}+b={BM25_B:g}'
    f'+lambda={MODEL1_LAMBDA:g}+probSelfTran={MODEL1_PROB_SELF_TRAN:g}'
)

extractor_json_rel = Path('exper_desc.best') / 'extractors' / f'{fid}.json'
extractor_json_abs = collect_dir / extractor_json_rel

extractor_json = [
    {
        'type': 'Model1Similarity',
        'params': {
            'queryFieldName': 'text',
            'indexFieldName': 'text',
            'gizaIterQty': '5',
            'probSelfTran': MODEL1_PROB_SELF_TRAN,
            'lambda': MODEL1_LAMBDA,
            'minModel1Prob': MODEL1_MIN_PROB
        }
    },
    {
        'type': 'TFIDFSimilarity',
        'params': {
            'queryFieldName': 'text',
            'indexFieldName': 'text',
            'similType': 'bm25',
            'k1': BM25_K1,
            'b': BM25_B
        }
    }
]

with extractor_json_abs.open('w', encoding='utf-8') as f:
    json.dump(extractor_json, f, indent=2)

final_desc_rel = Path('exper_desc.best') / 'bm25_model1_optimal.json'
final_desc_abs = collect_dir / final_desc_rel

final_desc = [
    {
        'experSubdir': 'feat_exper/bm25_model1_optimal',
        'extrTypeFinal': str(extractor_json_rel).replace('\\', '/'),
        'testOnly': 0
    }
]

with final_desc_abs.open('w', encoding='utf-8') as f:
    json.dump(final_desc, f, indent=2)

print('Created extractor JSON:', extractor_json_abs)
print('Created experiment descriptor:', final_desc_abs)
print()

# ============================================================================
# TRAIN + EVALUATE FINAL OPTIMAL FUSION
# ============================================================================

print('=' * 80)
print('TRAIN + EVALUATE FINAL OPTIMAL FUSION')
print('=' * 80)

cmd = [
    'bash', './exper/run_experiments.sh',
    COLLECT_NAME,
    str(final_desc_rel).replace('\\', '/'),
    '-test_part', TEST_PART,
    '-train_part', TRAIN_PART,
    '-train_cand_qty', str(TRAIN_CAND_QTY),
    '-model1_subdir', 'giza'
]

print('Running:', ' '.join(cmd))
print()
res = subprocess.run(cmd, cwd=SCRIPTS_DIR, env=os.environ.copy(), text=True, capture_output=True)
print(res.stdout)
if res.returncode != 0:
    print('STDERR:', res.stderr)
    sys.exit(f'Final BM25+Model1 experiment failed with code {res.returncode}')

print()

# ============================================================================
# SHOW FINAL REPORT
# ============================================================================

print('=' * 80)
print('SHOW FINAL REPORT')
print('=' * 80)

report_file = collect_dir / 'results' / TEST_PART / 'feat_exper' / 'bm25_model1_optimal' / 'rep' / 'out_100.rep'

if not report_file.exists():
    print(f'ERROR: Report not found: {report_file}')
    sys.exit(1)

print('Report file:', report_file)
print()
print(report_file.read_text(encoding='utf-8'))
print()

print('=' * 80)
print('DONE')
print('=' * 80)
