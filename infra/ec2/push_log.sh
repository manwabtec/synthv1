#!/usr/bin/env bash
# Commit one EC2 check log folder and push it, so it can be pulled on the laptop.
#
# Usage (on the EC2 instance, inside the repo):
#   bash infra/ec2/push_log.sh                 # newest folder in logs/ec2/
#   bash infra/ec2/push_log.sh <run_id|path>   # a specific one, e.g. 20261105-221530Z_c951053
#
# Only the log folder is committed (nothing else in the working tree is touched).
# The commit message names the run (UTC time + code commit) and its overall result.
# Pushing needs write access from this instance (e.g. a deploy key with write access).

set -euo pipefail

CODE_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$CODE_ROOT"

if [ $# -ge 1 ]; then
  RUN="$(basename "$1")"
else
  RUN="$(ls -1 logs/ec2 2>/dev/null | sort | tail -1 || true)"
fi
DIR="logs/ec2/$RUN"
[ -n "$RUN" ] && [ -d "$DIR" ] || { echo "No log folder found ($DIR). Run infra/ec2/check_ec2.sh first."; exit 1; }

RESULT="$(grep -m1 '^OVERALL:' "$DIR/summary.txt" 2>/dev/null | sed 's/^OVERALL: //' || true)"
RESULT="${RESULT:-incomplete run}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# Fallback identity for this commit only, if git user is not configured on the instance.
ID_ARGS=()
git config user.name  >/dev/null || ID_ARGS+=(-c "user.name=synthrig-ec2")
git config user.email >/dev/null || ID_ARGS+=(-c "user.email=synthrig-ec2@users.noreply.github.com")

git add -- "$DIR"
if git diff --cached --quiet -- "$DIR"; then
  echo "Nothing new to commit in $DIR (already pushed?)."
else
  git "${ID_ARGS[@]}" commit -q -m "ec2 log: $RUN" -m "$RESULT" \
      -m "Host: $(hostname). Created by infra/ec2/check_ec2.sh." -- "$DIR"
  echo "Committed $DIR"
fi

# Get any new laptop commits first; --autostash keeps unrelated local edits safe.
git pull --rebase --autostash -q origin "$BRANCH"
git push origin "HEAD:$BRANCH"
echo
echo "Pushed to origin/$BRANCH. On the laptop: git pull, then open $DIR/summary.txt"
