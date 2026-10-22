#!/usr/bin/env bash
# Copy the repo (code only) to /opt/dandori/app on the VM and install it into the venv.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
source "$HERE/env.local.sh"
VM="${VM:-pai2026-dandori}"
TAR=$(mktemp).tgz
COPYFILE_DISABLE=1 tar czf "$TAR" -C "$REPO" --exclude=.venv --exclude=outputs --exclude=.env --exclude='.env.*' --exclude=.git --exclude='._*' .
gcloud compute scp "$TAR" "$VM:/tmp/dandori_app.tgz" --project="$PROJECT" --zone="$ZONE" --tunnel-through-iap >/dev/null
gcloud compute ssh "$VM" --project="$PROJECT" --zone="$ZONE" --tunnel-through-iap --command \
  'sudo mkdir -p /opt/dandori/app && sudo tar xzf /tmp/dandori_app.tgz -C /opt/dandori/app && sudo chmod -R a+rwX /opt/dandori && sudo chown -R $USER /opt/dandori/app && rm -f /tmp/dandori_app.tgz && /opt/dandori/venv/bin/pip install -q -e /opt/dandori/app --no-deps && echo SYNC_OK' 2>&1 | grep -E "SYNC_OK|rror"
