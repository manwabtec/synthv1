#!/usr/bin/env bash
# Container entrypoint. Known commands run as-is, so both of these work:
#   docker run <image> pytest -q
#   docker run <image> --config configs/debug_clutter/run.yaml --run-dir ...   (-> infra/worker.py)
set -e
case "${1:-}" in
  pytest|python|python3|bash|sh|nvidia-smi|streamlit) exec "$@" ;;
  *) exec python /code/infra/worker.py "$@" ;;
esac
