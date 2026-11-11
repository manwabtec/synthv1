#!/usr/bin/env bash
# One-time (and after every stop/start) host setup for the synthrig EC2 instance.
#
# Usage (on the instance, needs sudo):
#   bash infra/ec2/setup_host.sh [--data-dir /app/synthrig] [--scratch /scratch]
#                                [--docker-root /app/docker] [--skip-docker]
#
# Idempotent. What it does:
#   1. Data dir on the persistent EBS volume (default /app/synthrig), owned by the
#      invoking user, with datasets/ previews/ logs/ cache/ subfolders.
#   2. Scratch: formats the EC2 *instance-store* NVMe (if it has no filesystem) and
#      mounts it at /scratch. Instance store is ERASED on every stop/start, so run
#      this script again after a restart. It never formats an EBS volume.
#   3. Docker: registers the NVIDIA container runtime and moves Docker's data-root
#      to the EBS volume (default /app/docker). Restarts the Docker daemon only if
#      something changed. The old /var/lib/docker is kept; delete it by hand.
# It never creates AWS resources.
set -euo pipefail

DATA_DIR=/app/synthrig
SCRATCH=/scratch
DOCKER_ROOT=/app/docker
SKIP_DOCKER=0
while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --scratch) SCRATCH="$2"; shift 2 ;;
    --docker-root) DOCKER_ROOT="$2"; shift 2 ;;
    --skip-docker) SKIP_DOCKER=1; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done
OWNER="${SUDO_USER:-$(id -un)}"
GROUP="$(id -gn "$OWNER")"

echo ">> data dir $DATA_DIR (owner $OWNER)"
sudo mkdir -p "$DATA_DIR"
sudo chown "$OWNER:$GROUP" "$DATA_DIR"
for d in datasets previews logs cache diagnostics; do mkdir -p "$DATA_DIR/$d" 2>/dev/null || sudo -u "$OWNER" mkdir -p "$DATA_DIR/$d"; done

echo ">> scratch $SCRATCH"
if mountpoint -q "$SCRATCH"; then
  echo "   already mounted: $(findmnt -no SOURCE "$SCRATCH")"
else
  DEV="$(lsblk -dpno NAME,MODEL | awk '/Instance Storage/ {print $1; exit}')"
  if [ -z "$DEV" ]; then
    echo "   no instance-store NVMe found; using $DATA_DIR/scratch on EBS instead"
    SCRATCH="$DATA_DIR/scratch"
  else
    if [ -z "$(sudo blkid -o value -s TYPE "$DEV" || true)" ]; then
      echo "   formatting instance store $DEV as xfs"
      sudo mkfs.xfs -q "$DEV"
    fi
    sudo mkdir -p "$SCRATCH"
    sudo mount -o noatime "$DEV" "$SCRATCH"
    echo "   mounted $DEV"
  fi
fi
sudo mkdir -p "$SCRATCH"
sudo chown "$OWNER:$GROUP" "$SCRATCH"

CONFIG="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/infra/config.yaml"
if [ ! -e "$CONFIG" ]; then
  echo ">> writing $CONFIG"
  printf '%s\n' "# Written by infra/ec2/setup_host.sh for the EC2 instance (gitignored)." \
    "data_root: $DATA_DIR" "scratch_dir: $SCRATCH" "docker_root: $DOCKER_ROOT" \
    | sudo -u "$OWNER" tee "$CONFIG" >/dev/null
else
  echo ">> $CONFIG exists; not changed"
fi

if [ "$SKIP_DOCKER" = 1 ]; then echo ">> docker: skipped"; exit 0; fi
echo ">> docker"
DAEMON=/etc/docker/daemon.json
BEFORE="$(sudo cat "$DAEMON" 2>/dev/null || echo '{}')"
sudo nvidia-ctk runtime configure --runtime=docker >/dev/null
sudo python3 - "$DAEMON" "$DOCKER_ROOT" <<'PY'
import json, sys
path, root = sys.argv[1], sys.argv[2]
with open(path) as f:
    cfg = json.load(f)
cfg["data-root"] = root
with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY
AFTER="$(sudo cat "$DAEMON")"
# Compare parsed JSON: nvidia-ctk and the writer above format the file differently.
if python3 -c 'import json,sys; sys.exit(json.loads(sys.argv[1]) != json.loads(sys.argv[2]))' "$BEFORE" "$AFTER"; then
  echo "   unchanged"
else
  OLD_ROOT="$(sudo docker info -f '{{.DockerRootDir}}')"
  echo "   restarting Docker (data-root $OLD_ROOT -> $DOCKER_ROOT)"
  sudo systemctl stop docker docker.socket
  if [ "$OLD_ROOT" != "$DOCKER_ROOT" ] && [ ! -e "$DOCKER_ROOT" ]; then
    sudo mkdir -p "$DOCKER_ROOT"
    sudo cp -a "$OLD_ROOT/." "$DOCKER_ROOT/"
  fi
  sudo systemctl start docker
fi
sudo docker info -f 'root={{.DockerRootDir}} driver={{.Driver}} runtimes={{range $k, $v := .Runtimes}}{{$k}} {{end}}'
echo ">> done. Data: $DATA_DIR  Scratch: $SCRATCH  Docker: $DOCKER_ROOT"
