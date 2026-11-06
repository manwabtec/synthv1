#!/usr/bin/env bash
# Check that this EC2 instance can run the synthrig pipeline, and log the results.
#
# Usage (from anywhere inside the repo, on the EC2 instance):
#   bash infra/ec2/check_ec2.sh [--pull] [--install-uv] [--quick]
#
#   --pull        git pull --ff-only first, then re-run the updated script
#   --install-uv  install uv into ~/.local/bin if it is missing (user-level, no sudo)
#   --quick       skip the demo render + preview (system, deps, tests, smoke renders only)
#
# Every step runs even if an earlier one fails (dependent steps are SKIPPED).
# Output: logs/ec2/<UTC-time>_<commit>/
#   summary.txt        one PASS/FAIL/SKIP line per step + timings  <- read this first
#   run_info.txt       branch, commit, host, time
#   NN_<step>.log      full output of each step
#   preview_app1.png   demo render preview (if the render succeeded)
# Then commit + push the log with:  bash infra/ec2/push_log.sh
#
# Read-only apart from: .venv/ (uv sync), ../data/diagnostics/ (renders), logs/ec2/.
# It never starts AWS resources and never prints credentials.

set -uo pipefail

PULL=0; INSTALL_UV=0; QUICK=0
for arg in "$@"; do
  case "$arg" in
    --pull) PULL=1 ;;
    --install-uv) INSTALL_UV=1 ;;
    --quick) QUICK=1 ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (see --help)"; exit 2 ;;
  esac
done

CODE_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null)" || {
  echo "Run this from inside the synthrig git checkout."; exit 2; }
cd "$CODE_ROOT"

# Pull first, then re-exec so the *updated* version of this script runs.
if [ "$PULL" = 1 ]; then
  echo ">> git pull --ff-only"
  git pull --ff-only || { echo "git pull failed; fix it (local changes? diverged branch?) and retry."; exit 1; }
  rest=(); for a in "$@"; do [ "$a" != "--pull" ] && rest+=("$a"); done
  exec bash "$CODE_ROOT/infra/ec2/check_ec2.sh" "${rest[@]}"
fi

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export NO_COLOR=1 UV_NO_PROGRESS=1 PYTHONUNBUFFERED=1

SHA="$(git rev-parse --short HEAD)"
RUN_ID="$(date -u +%Y%m%d-%H%M%SZ)_${SHA}"
LOG_DIR="$CODE_ROOT/logs/ec2/$RUN_ID"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.txt"
STEP_NO=0
declare -A RESULT

{
  echo "run_id:   $RUN_ID"
  echo "time_utc: $(date -u '+%Y-%m-%d %H:%M:%S')"
  echo "host:     $(hostname)"
  echo "branch:   $(git rev-parse --abbrev-ref HEAD)"
  echo "commit:   $(git rev-parse HEAD)"
  echo "subject:  $(git log -1 --format=%s)"
  echo "dirty:    $(git status --porcelain --untracked-files=no | wc -l) modified tracked file(s)"
  echo "options:  quick=$QUICK install_uv=$INSTALL_UV"
} > "$LOG_DIR/run_info.txt"
cat "$LOG_DIR/run_info.txt"
printf '%-22s %-5s %8s\n' STEP RESULT SECONDS > "$SUMMARY"

# step <name> <depends-on (space separated, or "")> <command...>
step() {
  local name="$1" deps="$2"; shift 2
  STEP_NO=$((STEP_NO + 1))
  local log; log="$LOG_DIR/$(printf '%02d' "$STEP_NO")_${name}.log"
  for d in $deps; do
    if [ "${RESULT[$d]:-}" != PASS ]; then
      RESULT[$name]=SKIP
      echo "skipped: dependency '$d' did not pass" > "$log"
      printf '%-22s %-5s %8s  (needs %s)\n' "$name" SKIP - "$d" | tee -a "$SUMMARY"
      return
    fi
  done
  echo; echo "======== [$STEP_NO] $name ========"
  local start=$SECONDS
  ( set -o pipefail; "$@" ) 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  local secs=$((SECONDS - start))
  if [ "$rc" = 0 ]; then RESULT[$name]=PASS; else RESULT[$name]=FAIL; fi
  printf '%-22s %-5s %8s\n' "$name" "${RESULT[$name]}" "$secs" | tee -a "$SUMMARY"
}

# ---------------------------------------------------------------- checks
sys_info() {
  echo "--- OS";      cat /etc/os-release 2>/dev/null | grep -E '^(PRETTY_NAME|VERSION_ID)='; uname -srm
  echo "--- CPU";     nproc; grep -m1 'model name' /proc/cpuinfo
  echo "--- Memory";  free -h
  echo "--- Disk (code + data)"; df -h "$CODE_ROOT" "$CODE_ROOT/.."
  echo "--- Instance (IMDSv2)"
  local tok; tok="$(curl -s -m 3 -X PUT http://169.254.169.254/latest/api/token \
                    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' || true)"
  if [ -n "$tok" ]; then
    for k in instance-id instance-type placement/availability-zone ami-id; do
      printf '%-28s %s\n' "$k" "$(curl -s -m 3 -H "X-aws-ec2-metadata-token: $tok" \
                                   "http://169.254.169.254/latest/meta-data/$k")"
    done
  else
    echo "metadata service not reachable (not EC2?)"
  fi
  local free_gb; free_gb="$(df -Pk "$CODE_ROOT" | awk 'NR==2 {print int($4/1048576)}')"
  echo "free disk: ${free_gb} GB (need >= 5 GB for .venv with Blender + renders)"
  [ "$free_gb" -ge 5 ]
}

gpu_driver() {
  echo "--- PCI NVIDIA devices"
  if command -v lspci >/dev/null; then lspci | grep -i nvidia || echo "(none found by lspci)"; else
    echo "(lspci not installed)"; fi
  echo "--- nvidia-smi"
  if ! command -v nvidia-smi >/dev/null; then
    echo "FAIL: nvidia-smi not found -> NVIDIA driver not installed (or not a GPU instance type)"; return 1; fi
  nvidia-smi
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
}

tools() {
  local ok=0
  for t in git curl uv aws; do
    if command -v "$t" >/dev/null; then printf '%-6s %s\n' "$t" "$("$t" --version 2>&1 | head -1)"
    else printf '%-6s MISSING\n' "$t"; [ "$t" = uv ] && ok=1; fi
  done
  echo "python3.11: $(command -v python3.11 || echo 'not on PATH (uv will download a managed 3.11)')"
  if [ "$ok" != 0 ]; then
    echo; echo "uv is required. Install (user-level, no sudo):"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh     # or re-run with --install-uv"
  fi
  return $ok
}

network() {
  local ok=0
  for url in https://github.com https://pypi.org/simple/bpy/ https://files.pythonhosted.org \
             https://github.com/astral-sh/python-build-standalone/releases; do
    code="$(curl -s -o /dev/null -m 15 -w '%{http_code}' "$url")"
    printf '%-62s HTTP %s\n' "$url" "$code"
    case "$code" in 2*|3*) ;; *) ok=1 ;; esac
  done
  echo "--- AWS identity of the instance role (no secrets)"
  aws sts get-caller-identity --query Arn --output text 2>&1 || echo "(no instance credentials or aws CLI missing)"
  return $ok
}

uv_sync() { uv sync --extra render; }

bpy_import() {
  uv run python - <<'PY'
import sys, bpy
print("python", sys.version.split()[0])
print("blender", bpy.app.version_string)
PY
  local rc=$?
  if [ $rc != 0 ]; then
    echo; echo "--- Missing shared libraries for bpy (install these with dnf):"
    find .venv -path '*site-packages/bpy*' -name '*.so*' -exec ldd {} \; 2>/dev/null \
      | grep 'not found' | sort -u || true
  fi
  return $rc
}

cycles_gpu() { uv run python -m synthrig.render.devices; }
pytests()    { uv run pytest -q -rs; }
smoke_cpu()  { uv run python infra/cloud.py local-check --render; }
smoke_gpu()  { uv run python infra/cloud.py local-check --gpu --render; }

demo_render() {
  local out="diagnostics/ec2_${RUN_ID}"
  uv run python run.py configs/runs/app1_demo.yaml --device GPU \
      --frames 2 --scale 0.25 --samples 16 --out "$out" || return 1
  uv run python eval/preview.py "../data/$out" --frame 0 --out "$LOG_DIR/preview_app1.png"
}

# ---------------------------------------------------------------- run
if [ "$INSTALL_UV" = 1 ] && ! command -v uv >/dev/null; then
  echo ">> installing uv to ~/.local/bin"; curl -LsSf https://astral.sh/uv/install.sh | sh
fi

step system       ""            sys_info
step gpu_driver   ""            gpu_driver
step tools        ""            tools
step network      ""            network
step uv_sync      "tools"       uv_sync
step bpy_import   "uv_sync"     bpy_import
step pytest       "uv_sync"     pytests
step cycles_gpu   "bpy_import gpu_driver" cycles_gpu
step smoke_cpu    "bpy_import"  smoke_cpu
step smoke_gpu    "cycles_gpu"  smoke_gpu
if [ "$QUICK" = 0 ]; then
  step demo_render "smoke_gpu"  demo_render
fi

FAILS=0
for r in "${RESULT[@]}"; do [ "$r" = PASS ] || FAILS=$((FAILS + 1)); done
{
  echo
  if [ "$FAILS" = 0 ]; then echo "OVERALL: PASS - pipeline requirements met on this instance"
  else echo "OVERALL: $FAILS step(s) failed or skipped - see the NN_<step>.log files"; fi
} | tee -a "$SUMMARY"

echo
echo "Log folder: logs/ec2/$RUN_ID"
echo "Send it back:  bash infra/ec2/push_log.sh"
[ "$FAILS" = 0 ]
