#!/usr/bin/env python3


# cd demo
# python 07_01_setup_and_export.py      # ~5-10 min
# python 07_02_start_nmslib_servers.py  # starts servers in background
# python 07_03_eval_bruteforce.py       # ~20-30 min (or run 04 in parallel in another shell)
# python 07_04_eval_napp.py             # ~20-30 min
# python 07_05_show_results.py          # <1 min, prints comparison table

"""
Step 1: Setup, load best params, build descriptors, and export sparse vectors.

Run from the flexneuart_scripts directory:
    cd demo
    python 07_01_setup_and_export.py
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
TEST_PART = 'test'
TRAIN_CAND_QTY = 1000
NMSLIB_CAND_PROV_QTY = 1000

BM25_TUNE_TOP_K = '250'
MODEL1_TUNE_TOP_K = '15'
SELECT_METRIC_COL = 'MAP'

NMSLIB_SERVER_HOST = os.environ.get('NMSLIB_SERVER_HOST', '127.0.0.1')
NMSLIB_BRUTEFORCE_PORT = int(os.environ.get('NMSLIB_BRUTEFORCE_PORT', '8000'))
NMSLIB_NAPP_PORT = int(os.environ.get('NMSLIB_NAPP_PORT', '8001'))
BRUTEFORCE_SERVER_URL = f'{NMSLIB_SERVER_HOST}:{NMSLIB_BRUTEFORCE_PORT}'
NAPP_SERVER_URL = f'{NMSLIB_SERVER_HOST}:{NMSLIB_NAPP_PORT}'

BM25_TUNE_TSV = SCRIPTS_DIR / 'bm25tune_stackoverflow_all.tsv'
MODEL1_TUNE_TSV = SCRIPTS_DIR / 'model1tune_stackoverflow_all.tsv'

collect_dir = COLLECT_ROOT / COLLECT_NAME
extractor_rel = Path('exper_desc.best') / 'extractors' / 'bm25_model1_optimal.json'
bruteforce_desc_rel = Path('exper_desc.best') / 'bm25_model1_optimal_bruteforce.json'
napp_desc_rel = Path('exper_desc.best') / 'bm25_model1_optimal_napp.json'

def resolve_nmslib_rel_dir():
    interleaved_dir = Path('nmslib') / 'bm25_model1_interleaved'
    legacy_dir = Path('nmslib') / 'bm25_model1'
    if (collect_dir / 'derived_data' / interleaved_dir).exists():
        return interleaved_dir
    if (collect_dir / 'derived_data' / legacy_dir).exists():
        return legacy_dir
    return interleaved_dir

nmslib_rel_dir = resolve_nmslib_rel_dir()
docs_export_rel = nmslib_rel_dir / 'docs_export.data'
queries_export_rel = nmslib_rel_dir / f'queries_{TEST_PART}_export.data'
docs_export_abs = collect_dir / 'derived_data' / docs_export_rel
queries_export_abs = collect_dir / 'derived_data' / queries_export_rel

RESUME_EXISTING_EXPORTS = os.environ.get('RESUME_EXISTING_EXPORTS', '0') == '1'

os.environ['COLLECT_ROOT'] = str(COLLECT_ROOT)

def subprocess_env():
    env = os.environ.copy()
    env['COLLECT_ROOT'] = str(COLLECT_ROOT)
    env['PYTHONPATH'] = str(REPO_ROOT) + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env

def run_cmd(cmd, cwd=SCRIPTS_DIR):
    print('Running:', ' '.join(str(x) for x in cmd))
    result = subprocess.run(cmd, cwd=cwd, env=subprocess_env(), text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print('STDERR:', result.stderr)
        raise RuntimeError(f'Command failed with code {result.returncode}')
    return result

print('=' * 80)
print('SETUP')
print('=' * 80)
print('REPO_ROOT             =', REPO_ROOT)
print('SCRIPTS_DIR           =', SCRIPTS_DIR)
print('COLLECT_ROOT          =', COLLECT_ROOT)
print('COLLECT_NAME          =', COLLECT_NAME)
print('TRAIN_PART            =', TRAIN_PART)
print('TEST_PART             =', TEST_PART)
print('BRUTEFORCE_SERVER_URL =', BRUTEFORCE_SERVER_URL)
print('NAPP_SERVER_URL       =', NAPP_SERVER_URL)
print()

# ============================================================================
# LOAD BEST PARAMS
# ============================================================================

print('=' * 80)
print('LOAD BEST PARAMS FROM TUNING TSVs')
print('=' * 80)

def load_best_row(tsv_path: Path, metric_col: str, top_k: str):
    with tsv_path.open('r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    if not rows:
        raise RuntimeError(f'No rows in TSV: {tsv_path}')
    filtered = [row for row in rows if row.get('top_k') == str(top_k)]
    if not filtered:
        raise RuntimeError(f'No rows with top_k={top_k} in {tsv_path}')
    best = max(filtered, key=lambda row: float(row[metric_col]))
    return best

def parse_bm25_from_exper_subdir(exper_subdir: str):
    match = re.search(r'k1=([0-9.]+)_b=([0-9.]+)', exper_subdir)
    if not match:
        raise RuntimeError(f'Cannot parse BM25 params from exper_subdir: {exper_subdir}')
    return float(match.group(1)), float(match.group(2))

def parse_model1_from_exper_subdir(exper_subdir: str):
    m_lambda = re.search(r'lambda=([0-9.]+)', exper_subdir)
    m_prob = re.search(r'probSelfTran=([0-9.]+)', exper_subdir)
    m_min = re.search(r'minTranProb=([0-9.eE+-]+)', exper_subdir)
    if not m_lambda or not m_prob:
        raise RuntimeError(f'Cannot parse Model1 params from exper_subdir: {exper_subdir}')
    return float(m_lambda.group(1)), float(m_prob.group(1)), float(m_min.group(1)) if m_min else 2.5e-3

def locate_trained_model():
    expected = collect_dir / 'results' / TEST_PART / 'feat_exper' / 'bm25_model1_optimal' / 'letor' / f'out_{COLLECT_NAME}_{TRAIN_PART}_{TRAIN_CAND_QTY}.model'
    if expected.exists():
        return expected
    matches = sorted((collect_dir / 'results').rglob(f'out_{COLLECT_NAME}_{TRAIN_PART}_{TRAIN_CAND_QTY}.model'))
    if not matches:
        raise FileNotFoundError(f'Could not find trained model file under {collect_dir / "results"}')
    return matches[0]

bm25_best = load_best_row(BM25_TUNE_TSV, SELECT_METRIC_COL, BM25_TUNE_TOP_K)
model1_best = load_best_row(MODEL1_TUNE_TSV, SELECT_METRIC_COL, MODEL1_TUNE_TOP_K)

BM25_K1, BM25_B = parse_bm25_from_exper_subdir(bm25_best['exper_subdir'])
MODEL1_LAMBDA, MODEL1_PROB_SELF_TRAN, MODEL1_MIN_PROB = parse_model1_from_exper_subdir(model1_best['exper_subdir'])
MODEL_FILE = locate_trained_model()
MODEL_FILE_REL = MODEL_FILE.relative_to(COLLECT_ROOT)
EXTRACTOR_JSON_ABS = collect_dir / extractor_rel

print('Best BM25 row (top_k=250):')
print(bm25_best)
print()
print('Best Model1 row (top_k=15):')
print(model1_best)
print()
print('Selected params:')
print('  BM25_K1 =', BM25_K1)
print('  BM25_B =', BM25_B)
print('  MODEL1_LAMBDA =', MODEL1_LAMBDA)
print('  MODEL1_PROB_SELF_TRAN =', MODEL1_PROB_SELF_TRAN)
print('  MODEL1_MIN_PROB =', MODEL1_MIN_PROB)
print()
print('Trained model file =', MODEL_FILE)
print('Extractor JSON     =', EXTRACTOR_JSON_ABS)
print()

# ============================================================================
# BUILD DESCRIPTORS
# ============================================================================

print('=' * 80)
print('BUILD DESCRIPTORS')
print('=' * 80)

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

bruteforce_desc = [
    {
        'experSubdir': 'feat_exper/bm25_model1_optimal_bruteforce',
        'extrTypeFinal': str(extractor_rel).replace('\\', '/'),
        'model_final': str(MODEL_FILE_REL).replace('\\', '/'),
        'cand_prov': 'nmslib',
        'cand_prov_uri': BRUTEFORCE_SERVER_URL,
        'cand_prov_add_conf': json.dumps({
            'extrType': str(extractor_rel).replace('\\', '/'),
            'sparseInterleave': True
        }),
        'cand_prov_qty': NMSLIB_CAND_PROV_QTY,
        'testOnly': 1
    }
]

napp_desc = [
    {
        'experSubdir': 'feat_exper/bm25_model1_optimal_napp',
        'extrTypeFinal': str(extractor_rel).replace('\\', '/'),
        'model_final': str(MODEL_FILE_REL).replace('\\', '/'),
        'cand_prov': 'nmslib',
        'cand_prov_uri': NAPP_SERVER_URL,
        'cand_prov_add_conf': json.dumps({
            'extrType': str(extractor_rel).replace('\\', '/'),
            'sparseInterleave': True
        }),
        'cand_prov_qty': NMSLIB_CAND_PROV_QTY,
        'testOnly': 1
    }
]

BRUTEFORCE_DESC_ABS = collect_dir / bruteforce_desc_rel
NAPP_DESC_ABS = collect_dir / napp_desc_rel

for path, payload in [
    (EXTRACTOR_JSON_ABS, extractor_json),
    (BRUTEFORCE_DESC_ABS, bruteforce_desc),
    (NAPP_DESC_ABS, napp_desc),
]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

print('Wrote extractor JSON:', EXTRACTOR_JSON_ABS)
print('Wrote brute-force descriptor:', BRUTEFORCE_DESC_ABS)
print('Wrote NAPP descriptor:', NAPP_DESC_ABS)
print()

# ============================================================================
# EXPORT SPARSE VECTORS
# ============================================================================

print('=' * 80)
print('EXPORT SPARSE VECTORS')
print('=' * 80)

cmd_docs = [
    'bash', './export_nmslib/export_nmslib_sparse.sh',
    COLLECT_NAME,
    str(extractor_rel).replace('\\', '/'),
    str(docs_export_rel).replace('\\', '/'),
    '-model_file',
    str(MODEL_FILE)
]

cmd_queries = [
    'bash', './export_nmslib/export_nmslib_sparse.sh',
    COLLECT_NAME,
    str(extractor_rel).replace('\\', '/'),
    str(queries_export_rel).replace('\\', '/'),
    '-query_file_pref',
    str(collect_dir / 'input_data' / TEST_PART / 'QuestionFields'),
    '-model_file',
    str(MODEL_FILE)
]

if not RESUME_EXISTING_EXPORTS or not docs_export_abs.exists():
    run_cmd(cmd_docs)
else:
    print('Skipping docs export, already exists:', docs_export_abs)

if not RESUME_EXISTING_EXPORTS or not queries_export_abs.exists():
    run_cmd(cmd_queries)
else:
    print('Skipping queries export, already exists:', queries_export_abs)

if not docs_export_abs.exists():
    raise FileNotFoundError(f'Docs export missing: {docs_export_abs}')
if not queries_export_abs.exists():
    raise FileNotFoundError(f'Queries export missing: {queries_export_abs}')

print('Created docs export   =', docs_export_abs)
print('Created queries export=', queries_export_abs)
print('Docs export size      =', docs_export_abs.stat().st_size)
print('Queries export size   =', queries_export_abs.stat().st_size)
print()

print('=' * 80)
print('STEP 1 COMPLETE')
print('=' * 80)
print('Next: run 07_02_start_nmslib_servers.py')
