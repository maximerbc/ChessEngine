#!/usr/bin/env bash
# Run manually only AFTER the user approves the concrete Runpod training quote.
set -euo pipefail
if [[ "${CHESS_POLICY_RUNPOD_APPROVED:-}" != "yes" ]]; then
  echo 'Runpod training is gated. Obtain user approval, then set CHESS_POLICY_RUNPOD_APPROVED=yes.' >&2
  exit 2
fi
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m chessengine.move_policy_train benchmark --device cuda
python -m chessengine.move_policy_train train --device cuda "$@"
python -m chessengine.move_policy_train evaluate --device cuda
