#!/usr/bin/env bash
# Run one command detached on the VM, mirroring status/log to GCS.
#   bash job.sh <job-name> <script> [args...]
# Env: GCS (required), SHUTDOWN=1 to power off when done.
set -uo pipefail
JOB="$1"; shift
: "${GCS:?}"
LOG="/var/tmp/dandori_${JOB}_$(date +%s).log"
push() { gcloud storage cp -q "$LOG" "$GCS/$JOB/log.txt" >/dev/null 2>&1 || true; }
status() { echo "$1 $(date -Is)" | gcloud storage cp -q - "$GCS/$JOB/status" >/dev/null 2>&1 || true; }
status RUNNING
( while true; do sleep 60; push; done ) & HB=$!
"$@" >>"$LOG" 2>&1
RC=$?
kill $HB; push
if [ $RC -eq 0 ]; then status DONE; else status "FAILED_$RC"; fi
[ "${SHUTDOWN:-0}" = 1 ] && sudo shutdown -h now
exit $RC
