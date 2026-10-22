#!/usr/bin/env bash
# Executor-only chain eval: given plan, oracle stop; with vs without return-to-home.
set -euo pipefail
cd /opt/dandori/app
PY=/opt/dandori/venv/bin/python
OUT=/opt/dandori/outputs/oracle_chains
$PY scripts/eval_chains.py --mode oracle --given-plan --episodes 10 --videos 2 --out $OUT/home
$PY scripts/eval_chains.py --mode oracle --given-plan --episodes 10 --videos 2 --no-home --out $OUT/nohome
gcloud storage cp -q -r $OUT "$GCS/oracle_chains/"
