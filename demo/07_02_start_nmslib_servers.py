#!/usr/bin/env python3

"""
Step 2: Start NMSLIB servers in brute_force and napp modes.

Run from the demo directory:
    cd demo
    python 07_02_start_nmslib_servers.py

This starts two background servers that continue to run.
"""

import os
import socket
import subprocess
import sys
import time
import shutil
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================

REPO_ROOT = Path('/tempory/the_three_potatoes/ri_project/workspaces/amelie/FlexNeuART-IR-TTP')
COLLECT_ROOT = (REPO_ROOT / 'demo' / 'collections').resolve()
COLLECT_NAME = 'stackoverflow_all'
TEST_PART = 'test'

NMSLIB_SERVER_HOST = os.environ.get('NMSLIB_SERVER_HOST', '127.0.0.1')
NMSLIB_BRUTEFORCE_PORT = int(os.environ.get('NMSLIB_BRUTEFORCE_PORT', '8000'))
NMSLIB_NAPP_PORT = int(os.environ.get('NMSLIB_NAPP_PORT', '8001'))

collect_dir = COLLECT_ROOT / COLLECT_NAME

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

def resolve_query_server_bin():
    candidates = []

    env_bin = os.environ.get('NMSLIB_QUERY_SERVER_BIN')
    if env_bin:
        candidates.append(Path(env_bin))

    candidates.extend([
        REPO_ROOT / 'query_server',
        REPO_ROOT / 'java' / 'query_server',
        REPO_ROOT / 'java' / 'target' / 'query_server',
        REPO_ROOT / 'java' / 'target' / 'bin' / 'query_server',
    ])

    which_bin = shutil.which('query_server')
    if which_bin:
        candidates.append(Path(which_bin))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return None

NMSLIB_QUERY_SERVER_BIN = resolve_query_server_bin()
NMSLIB_SERVER_SIMILARITY = 'negdotprod_sparse_fast'
NMSLIB_SERVER_INDEX_FILE = collect_dir / 'derived_data' / docs_export_rel

def subprocess_env():
    env = os.environ.copy()
    env['COLLECT_ROOT'] = str(COLLECT_ROOT)
    return env

print('=' * 80)
print('START NMSLIB SERVERS')
print('=' * 80)
print('NMSLIB_QUERY_SERVER_BIN    =', NMSLIB_QUERY_SERVER_BIN)
print('NMSLIB_SERVER_INDEX_FILE   =', NMSLIB_SERVER_INDEX_FILE)
print('NMSLIB_SERVER_SIMILARITY   =', NMSLIB_SERVER_SIMILARITY)
print('NMSLIB_SERVER_HOST         =', NMSLIB_SERVER_HOST)
print('NMSLIB_BRUTEFORCE_PORT     =', NMSLIB_BRUTEFORCE_PORT)
print('NMSLIB_NAPP_PORT           =', NMSLIB_NAPP_PORT)
print()

if NMSLIB_QUERY_SERVER_BIN is None:
    print('ERROR: NMSLIB query server binary not found.')
    print('Set NMSLIB_QUERY_SERVER_BIN to the built query_server executable.')
    print('Checked:')
    print('  - $NMSLIB_QUERY_SERVER_BIN')
    print('  - query_server on PATH')
    print('  -', REPO_ROOT / 'query_server')
    print('  -', REPO_ROOT / 'java' / 'query_server')
    print('  -', REPO_ROOT / 'java' / 'target' / 'query_server')
    print('  -', REPO_ROOT / 'java' / 'target' / 'bin' / 'query_server')
    sys.exit(1)

if not NMSLIB_SERVER_INDEX_FILE.exists():
    print(f'ERROR: NMSLIB index file not found: {NMSLIB_SERVER_INDEX_FILE}')
    print('Run 07_01_setup_and_export.py first.')
    sys.exit(1)

def start_nmslib_server(mode: str, port: int):
    url = f'{NMSLIB_SERVER_HOST}:{port}'
    log_file = collect_dir / 'results' / TEST_PART / f'nmslib_server_{mode}.log'
    pid_file = collect_dir / 'results' / TEST_PART / f'nmslib_server_{mode}.pid'

    server_cmd = [
        str(NMSLIB_QUERY_SERVER_BIN),
        '-p', str(port),
        '-s', NMSLIB_SERVER_SIMILARITY,
        '-i', str(NMSLIB_SERVER_INDEX_FILE),
        '-m', mode,
    ]

    log_file.parent.mkdir(parents=True, exist_ok=True)
    if pid_file.exists():
        existing_pid = pid_file.read_text(encoding='utf-8').strip()
        if existing_pid:
            try:
                os.kill(int(existing_pid), 0)
                print(f'NMSLIB {mode} server already running with PID {existing_pid} on {url}')
                return url
            except OSError:
                pid_file.unlink(missing_ok=True)

    log_handle = log_file.open('a', encoding='utf-8')
    process = subprocess.Popen(
        server_cmd,
        cwd=REPO_ROOT / 'demo',
        stdout=log_handle,
        stderr=log_handle,
        env=subprocess_env(),
    )
    pid_file.write_text(str(process.pid), encoding='utf-8')
    print(f'Started NMSLIB {mode} server:')
    print('  PID =', process.pid)
    print('  URL =', url)
    print('  LOG =', log_file)

    for _ in range(60):
        try:
            with socket.create_connection((NMSLIB_SERVER_HOST, port), timeout=1):
                print(f'NMSLIB {mode} server is accepting connections.')
                break
        except OSError:
            time.sleep(1)
    else:
        raise RuntimeError(f'NMSLIB {mode} server did not start within timeout. Check {log_file}')

    return url

bruteforce_url = start_nmslib_server('brute_force', NMSLIB_BRUTEFORCE_PORT)
print()
napp_url = start_nmslib_server('napp', NMSLIB_NAPP_PORT)
print()

print('=' * 80)
print('SERVERS STARTED')
print('=' * 80)
print('Brute-force URL:', bruteforce_url)
print('NAPP URL:      ', napp_url)
print()
print('Next: run 07_03_eval_bruteforce.py')
print('      (and/or 07_04_eval_napp.py in parallel or after)')
