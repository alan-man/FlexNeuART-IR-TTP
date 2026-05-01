#!/usr/bin/env bash
set -euo pipefail
cd /tempory/the_three_potatoes/ri_project/workspaces/amelie/FlexNeuART-IR-TTP/demo

if [ -z "${NMSLIB_QUERY_SERVER_BIN:-}" ]; then
  default_query_server_bin="/tempory/the_three_potatoes/ri_project/workspaces/nmslib/query_server/cpp_client_server/query_server"
  if [ -x "$default_query_server_bin" ]; then
    export NMSLIB_QUERY_SERVER_BIN="$default_query_server_bin"
    echo "[INFO] Using NMSLIB_QUERY_SERVER_BIN=$NMSLIB_QUERY_SERVER_BIN"
  fi
fi

run() {
  local name="$1"; shift
  echo "[START] $name"
  if ! python "$name" "$@"; then
    echo "[FAIL]  $name exited with $?" >&2
    exit 1
  fi
  echo "[DONE]  $name"
}

# run 07_01_setup_and_export.py
# run 07_02_start_nmslib_servers.py

# steps 3 & 4 in parallel
run 07_03_eval_bruteforce.py &
PID3=$!
run 07_04_eval_napp.py &
PID4=$!

FAIL=0
wait $PID3 || FAIL=$?
wait $PID4 || FAIL=$?
if [ $FAIL -ne 0 ]; then
  echo "[FAIL]  parallel step failed (exit $FAIL)" >&2
  exit $FAIL
fi

run 07_05_show_results.py
echo "[DONE]  all steps complete"