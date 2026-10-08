#!/usr/bin/env bash
set -euo pipefail

# Edit these defaults, or override any of them from the shell:
#   WORKERS=16 LIMIT=10 bash examples/run_batch.sh
# Extra arguments are passed through to examples/main.py:
#   bash examples/run_batch.sh --start-index 20 --fail-fast

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-${REPO_ROOT}/config/config.yaml}"
TASKS="${TASKS:-${SCRIPT_DIR}/test_data/test.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/vlm_annotation_batch_predictions}"
SUMMARY="${SUMMARY:-${SCRIPT_DIR}/trajectory.json}"

WORKERS="${WORKERS:-4}"
START_INDEX="${START_INDEX:-0}"
LIMIT="${LIMIT:-}"
START_FROM="${START_FROM:-}"
STOP_AFTER="${STOP_AFTER:-}"

DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
FAIL_FAST="${FAIL_FAST:-0}"

cmd=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/main.py"
  --config "${CONFIG}"
  --tasks "${TASKS}"
  --output-dir "${OUTPUT_DIR}"
  --summary "${SUMMARY}"
  --workers "${WORKERS}"
  --start-index "${START_INDEX}"
)

if [[ -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi

if [[ -n "${START_FROM}" ]]; then
  cmd+=(--start-from "${START_FROM}")
fi

if [[ -n "${STOP_AFTER}" ]]; then
  cmd+=(--stop-after "${STOP_AFTER}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  cmd+=(--skip-existing)
fi

if [[ "${FAIL_FAST}" == "1" ]]; then
  cmd+=(--fail-fast)
fi

cmd+=("$@")

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

cd "${REPO_ROOT}"
exec "${cmd[@]}"
