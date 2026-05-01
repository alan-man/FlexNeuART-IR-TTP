#!/usr/bin/env python3

"""
Step 3: Run brute-force evaluation (test-only).

Run from the demo directory after starting servers:
    cd demo
    python 07_03_eval_bruteforce.py
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
NMSLIB_BRUTEFORCE_PORT = int(os.environ.get('NMSLIB_BRUTEFORCE_PORT', '8000'))
BRUTEFORCE_SERVER_URL = f'{NMSLIB_SERVER_HOST}:{NMSLIB_BRUTEFORCE_PORT}'

collect_dir = COLLECT_ROOT / COLLECT_NAME
bruteforce_desc_rel = Path('exper_desc.best') / 'bm25_model1_optimal_bruteforce.json'

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
print('RUN BRUTE-FORCE EVALUATION')
print('=' * 80)
print('BRUTEFORCE_SERVER_URL =', BRUTEFORCE_SERVER_URL)
print()

print('Running NMSLIB brute-force test-only evaluation...')
cmd_bruteforce = [
    'bash', './exper/run_experiments.sh',
    COLLECT_NAME,
    str(bruteforce_desc_rel).replace('\\', '/'),
    '-test_part', TEST_PART,
    '-train_part', TRAIN_PART
]
run_cmd(cmd_bruteforce)

bruteforce_rep_dir = collect_dir / 'results' / TEST_PART / 'feat_exper' / 'bm25_model1_optimal_bruteforce' / 'rep'
if not bruteforce_rep_dir.exists():
    print(f'ERROR: Brute-force report directory not found: {bruteforce_rep_dir}')
    sys.exit(1)

print()
print('=' * 80)
print('BRUTE-FORCE EVALUATION COMPLETE')
print('=' * 80)
print('Brute-force report directory =', bruteforce_rep_dir)
print()
print('Next: run 07_04_eval_napp.py (if not already running)')
print('      then run 07_05_show_results.py')
