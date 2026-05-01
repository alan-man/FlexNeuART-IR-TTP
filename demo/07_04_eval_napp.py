#!/usr/bin/env python3

"""
Step 4: Run NAPP evaluation (test-only).

Run from the demo directory after starting servers:
    cd demo
    python 07_04_eval_napp.py
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path('/tempory/the_three_potatoes/ri_project/workspaces/amelie/FlexNeuART-IR-TTP')
SCRIPTS_DIR = REPO_ROOT / 'demo' / 'flexneuart_scripts'
COLLECT_ROOT = (REPO_ROOT / 'demo' / 'collections').resolve()
COLLECT_NAME = 'stackoverflow_all'
TEST_PART = 'test'
TRAIN_PART = 'train'

NMSLIB_SERVER_HOST = os.environ.get('NMSLIB_SERVER_HOST', '127.0.0.1')
NMSLIB_NAPP_PORT = int(os.environ.get('NMSLIB_NAPP_PORT', '8001'))
NAPP_SERVER_URL = f'{NMSLIB_SERVER_HOST}:{NMSLIB_NAPP_PORT}'

collect_dir = COLLECT_ROOT / COLLECT_NAME
napp_desc_rel = Path('exper_desc.best') / 'bm25_model1_optimal_napp.json'

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
print('RUN NAPP EVALUATION')
print('=' * 80)
print('NAPP_SERVER_URL =', NAPP_SERVER_URL)
print()

print('Running NMSLIB NAPP test-only evaluation...')
cmd_napp = [
    'bash', './exper/run_experiments.sh',
    COLLECT_NAME,
    str(napp_desc_rel).replace('\\', '/'),
    '-test_part', TEST_PART,
    '-train_part', TRAIN_PART
]
run_cmd(cmd_napp)

napp_rep_dir = collect_dir / 'results' / TEST_PART / 'feat_exper' / 'bm25_model1_optimal_napp' / 'rep'
if not napp_rep_dir.exists():
    print(f'ERROR: NAPP report directory not found: {napp_rep_dir}')
    sys.exit(1)

print()
print('=' * 80)
print('NAPP EVALUATION COMPLETE')
print('=' * 80)
print('NAPP report directory =', napp_rep_dir)
print()
print('Next: run 07_05_show_results.py')
