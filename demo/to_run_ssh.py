import os
import subprocess
from pathlib import Path

REPO_ROOT = Path('/tempory/the_three_potatoes/ri_project/workspaces/amelie/FlexNeuART-IR-TTP')
SCRIPTS_DIR = REPO_ROOT / 'demo' / 'flexneuart_scripts'
COLLECT_ROOT = (REPO_ROOT / 'demo' / 'collections').resolve()
# This must be a collection that contains both `input_data/train` and `input_data/dev1`.
COLLECT_NAME = 'stackoverflow_all'

# Train the fusion model on `train` and evaluate it on `dev1`.
TRAIN_PART = 'train'
TEST_PART = 'dev1'
TRAIN_CAND_QTY = "15,1,2,3,4,5,10,20,25,30,35,45,50,60,70,80,90,100"

# Field choices for StackOverflow conversion notebook: text, text_unlemm, text_raw, bigram
BM25_INDEX_FIELD = 'text'
BM25_QUERY_FIELD = 'text'
MODEL1_INDEX_FIELD = 'text'
MODEL1_QUERY_FIELD = 'text'

env = os.environ.copy()
    
os.environ['COLLECT_ROOT'] = str(COLLECT_ROOT)
env['PYTHONPATH'] = str(REPO_ROOT) + ':' + env.get('PYTHONPATH', '')
env['COLLECT_ROOT'] = str(COLLECT_ROOT)

collect_dir = COLLECT_ROOT / COLLECT_NAME
exper_desc_dir = collect_dir / 'exper_desc'
exper_desc_dir.mkdir(parents=True, exist_ok=True)

print('REPO_ROOT    =', REPO_ROOT)
print('SCRIPTS_DIR  =', SCRIPTS_DIR)
print('COLLECT_ROOT =', COLLECT_ROOT)
print('COLLECT_NAME =', COLLECT_NAME)
print('TRAIN_PART   =', TRAIN_PART)
print('TEST_PART    =', TEST_PART)
print('TRAIN_CAND_QTY =', TRAIN_CAND_QTY)
print('EXPER_DESC   =', exper_desc_dir)

required_paths = [
    collect_dir / 'input_data' / TRAIN_PART / 'QuestionFields.jsonl',
    collect_dir / 'input_data' / TRAIN_PART / 'AnswerFields.jsonl',
    collect_dir / 'input_data' / TRAIN_PART / 'qrels.txt',
    collect_dir / 'input_data' / TEST_PART / 'QuestionFields.jsonl',
    collect_dir / 'input_data' / TEST_PART / 'AnswerFields.jsonl',
    collect_dir / 'input_data' / TEST_PART / 'qrels.txt',
    collect_dir / 'lucene_index',
    collect_dir / 'forward_index' / BM25_INDEX_FIELD,
    collect_dir / 'derived_data' / 'giza' / f'{MODEL1_INDEX_FIELD}' / 'output.t1.5.bin'
]

for p in required_paths:
    if not p.exists():
        raise FileNotFoundError(f'Missing prerequisite: {p}')

print('All prerequisites found.')



cmd = [
    'python3', './gen_exper_desc/gen_bm25_tune_json_desc.py',
    '--index_field_name', BM25_INDEX_FIELD,
    '--query_field_name', BM25_QUERY_FIELD,
    '--outdir', str(exper_desc_dir),
    '--exper_subdir', 'tuning',
    '--rel_desc_path', 'exper_desc'
]
print('Running:', ' '.join(cmd))
res = subprocess.run(cmd, cwd=SCRIPTS_DIR, env=os.environ.copy(), text=True, capture_output=True)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr)
    raise RuntimeError(f'BM25 descriptor generation failed: {res.returncode}')

bm25_desc = f'exper_desc/bm25tune_{BM25_QUERY_FIELD}_{BM25_INDEX_FIELD}.json'
cmd = [
    'bash', './exper/run_experiments.sh',
    COLLECT_NAME,
    bm25_desc,
    '-test_part', TEST_PART,
    '-train_part', TRAIN_PART,
    '-train_cand_qty', str(TRAIN_CAND_QTY),
    '-early_stop_enabled', '0',           # Enable scanning all configs
    '-early_stop_threshold', '0',         # Strict: new must beat best
    '-early_stop_at_n', '100'              # Check at N=15 only
]
print('Running:', ' '.join(cmd))
res = subprocess.run(cmd, cwd=SCRIPTS_DIR, env=os.environ.copy(), text=True, capture_output=True)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr)
    raise RuntimeError(f'BM25 tuning experiments failed: {res.returncode}')