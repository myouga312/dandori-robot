#!/usr/bin/env bash
# Manage the project's GCE VM (L4).  Always stop the VM after use.
#   bash scripts/vm/vm.sh create [startup-script]
#   bash scripts/vm/vm.sh start|stop|status|ssh
#   bash scripts/vm/vm.sh log <job>      # tail job log from GCS
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# PROJECT / GCS etc. live in an untracked file (see env.local.sh.example)
[ -f "$HERE/env.local.sh" ] && source "$HERE/env.local.sh"
PROJECT="${PROJECT:?set PROJECT in scripts/vm/env.local.sh}"
GCS="${GCS:?set GCS in scripts/vm/env.local.sh}"
ZONE="${ZONE:-asia-northeast1-a}"
VM="${VM:-pai2026-dandori}"

# Bake the GCS prefix into a temp copy of a startup script.
render() { local out; out=$(mktemp); sed "s|@@GCS@@|$GCS|g" "$1" > "$out"; echo "$out"; }

case "${1:?create|start|stop|status|ssh|log}" in
create)
    gcloud compute instances create "$VM" --project="$PROJECT" --zone="$ZONE" \
        --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 \
        --image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
        --boot-disk-size=150GB --boot-disk-type=pd-ssd --maintenance-policy=TERMINATE \
        --scopes=cloud-platform \
        --metadata="install-nvidia-driver=True" \
        --metadata-from-file=startup-script="$(render "${2:-$HERE/startup_setup.sh}")"
    ;;
startup)  # replace startup script, then (re)boot:  vm.sh startup <script>
    gcloud compute instances add-metadata "$VM" --project="$PROJECT" --zone="$ZONE" \
        --metadata-from-file=startup-script="$(render "${2:?script}")"
    ;;
start)  gcloud compute instances start "$VM" --project="$PROJECT" --zone="$ZONE" ;;
stop)   gcloud compute instances stop "$VM" --project="$PROJECT" --zone="$ZONE" ;;
status)
    gcloud compute instances describe "$VM" --project="$PROJECT" --zone="$ZONE" --format="value(status)"
    gcloud storage cat "$GCS/*/status" 2>/dev/null || true
    ;;
ssh)    shift; gcloud compute ssh "$VM" --project="$PROJECT" --zone="$ZONE" --tunnel-through-iap "$@" ;;
log)    gcloud storage cat "$GCS/${2:?job}/log.txt" | tail -n "${3:-40}" ;;
esac
