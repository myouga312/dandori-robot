#!/usr/bin/env bash
# GCE startup-script: build the sim environment and run a smoke eval, then power off.
# Runs as root on every boot; the marker file makes it one-shot.
# Logs go to $GCS/<job>/ so progress is visible without SSH.
set -uo pipefail

GCS="@@GCS@@"
JOB="setup_smoke"
MARK=/var/tmp/dandori_${JOB}.done
ROOT=/opt/dandori
LOG=/var/tmp/dandori_${JOB}.log

[ -f "$MARK" ] && exit 0
touch "$MARK"

push() { gcloud storage cp -q "$LOG" "$GCS/$JOB/log.txt" >/dev/null 2>&1 || true; }
status() { echo "$1 $(date -Is)" | gcloud storage cp -q - "$GCS/$JOB/status" >/dev/null 2>&1 || true; }
step() { echo "=== $(date -Is) $*" | tee -a "$LOG"; push; }

status RUNNING
exec >>"$LOG" 2>&1
( while true; do sleep 120; push; done ) &

step "nvidia-smi"
nvidia-smi || { status FAILED_nvidia; push; exit 1; }

step "apt"
add-apt-repository -y ppa:deadsnakes/ppa || true
apt-get update -q
apt-get install -y -q python3.12 python3.12-venv python3.12-dev ffmpeg libegl1 libgl1 libosmesa6 cmake build-essential fonts-noto-cjk
# The DL image ships the compute-only driver; headless MuJoCo needs NVIDIA's EGL vendor library
# of the *same* version, otherwise glvnd falls back to Mesa ("failed to create dri2 screen").
DRV=$(dpkg-query -W -f='${Version}' libnvidia-compute-580-server)
apt-get install -y -q "libnvidia-gl-580-server=$DRV"

step "venv + lerobot"
mkdir -p $ROOT && cd $ROOT
python3.12 -m venv venv
venv/bin/pip install -q --upgrade pip
venv/bin/pip install "lerobot[smolvla,libero]==0.6.0" || venv/bin/pip install "lerobot[smolvla]==0.6.0" hf-libero
venv/bin/pip install -q "anthropic>=1.0" python-dotenv "imageio[ffmpeg]" av
venv/bin/pip freeze > $ROOT/pip_freeze.txt
gcloud storage cp -q $ROOT/pip_freeze.txt "$GCS/$JOB/pip_freeze.txt"

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HOME=$ROOT/hf

step "libero first import (writes ~/.libero/config.yaml)"
printf 'N\n' | venv/bin/python -c "from libero.libero import benchmark; b=benchmark.get_benchmark_dict()['libero_goal'](); print([b.get_task(i).language for i in range(b.n_tasks)])"

step "download policy"
venv/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('HuggingFaceVLA/smolvla_libero'))"

step "smoke eval: libero_goal, n_action_steps=10"
venv/bin/lerobot-eval \
    --policy.path=HuggingFaceVLA/smolvla_libero \
    --policy.n_action_steps=10 \
    --env.type=libero --env.task=libero_goal \
    --eval.batch_size=1 --eval.n_episodes=3 \
    --output_dir=$ROOT/outputs/smoke_goal
RC=$?
gcloud storage cp -q -r $ROOT/outputs/smoke_goal "$GCS/$JOB/" || true

chmod -R a+rwX $ROOT /root/.libero 2>/dev/null || true
if [ $RC -eq 0 ]; then status DONE; else status "FAILED_eval_$RC"; fi
push
shutdown -h now
