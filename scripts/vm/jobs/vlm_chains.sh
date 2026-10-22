#!/usr/bin/env bash
# Full system eval: Claude Code planner + verifier (VLM stop / retry / replan), home + grace.
# Every verifier call is dumped with simulator ground truth for the offline Claude vs GPT-6 comparison.
set -euo pipefail
cd /opt/dandori/app
# LLM calls are relayed through GCS to scripts/relay_worker.py running where Claude Code is logged in.
export LLM_BACKEND=relay RELAY_GCS="$GCS/relay" RELAY_BACKEND=${RELAY_BACKEND:-claude-cli}
OUT=/opt/dandori/outputs/vlm_chains
export VERIFY_DUMP_DIR=$OUT/verifier_dump
/opt/dandori/venv/bin/python scripts/eval_chains.py --mode vlm --episodes ${EPISODES:-10} --scenarios ${SCENARIOS:-S1,S2,S3,S4,S5} --videos 2 --out $OUT
gcloud storage cp -q -r $OUT "$GCS/vlm_chains/"
