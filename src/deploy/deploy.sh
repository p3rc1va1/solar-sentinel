#!/usr/bin/env bash
# =============================================================================
# Solar Sentinel — Deploy script (runs on the Pi via the self-hosted runner)
#
# What it does, in order:
#   1. rsync the freshly-checked-out `src/` tree into $DEPLOY_TARGET (default
#      /opt/solar-sentinel), preserving runtime data (DB, captures, model).
#   2. uv sync --inexact in the target so locked deps match pyproject, but
#      picamera2 (installed separately on the Pi) is NOT pruned.
#   3. sudo systemctl restart solar-sentinel  (passwordless via sudoers.d).
#   4. Poll /health until 200 OK or timeout — fail loud if the new code
#      doesn't come up cleanly.
#
# Runs as the runner user (e.g. `bahapie`). That user must:
#   - own $DEPLOY_TARGET (and have rsync/uv on PATH)
#   - have a sudoers rule for `systemctl restart|status solar-sentinel`
# =============================================================================

set -euo pipefail

DEPLOY_TARGET="${DEPLOY_TARGET:-/opt/solar-sentinel}"
HEALTH_URL="${HEALTH_URL:-http://localhost:8000/health}"
SERVICE_NAME="solar-sentinel"
HEALTH_TRIES=15
HEALTH_SLEEP=2

# The runner checks out the whole repo; we only deploy the `src/` subtree.
SOURCE_DIR="${GITHUB_WORKSPACE:-$(pwd)}/src/"

echo "=== Solar Sentinel deploy ==="
echo "  source : $SOURCE_DIR"
echo "  target : $DEPLOY_TARGET"
echo

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "ERROR: source dir not found: $SOURCE_DIR" >&2
  exit 1
fi

if [[ ! -d "$DEPLOY_TARGET" ]]; then
  echo "ERROR: target dir not found: $DEPLOY_TARGET" >&2
  echo "Bootstrap it first (see src/deploy/README.md)." >&2
  exit 1
fi

# 1. Sync code. --delete keeps the target clean of removed files, but excludes
#    preserve runtime state and the runner-side .git directory.
echo "[1/4] rsync code into $DEPLOY_TARGET"
rsync -a --delete \
  --exclude='data/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='.git/' \
  --exclude='*.pyc' \
  "$SOURCE_DIR" "$DEPLOY_TARGET/"

# 2. Sync deps. --inexact = "match the lock, but don't prune extras I installed
#    by hand" — preserves picamera2 which lives outside pyproject.
echo "[2/4] uv sync (inexact, keeps picamera2)"
cd "$DEPLOY_TARGET"
uv sync --inexact

# 3. Restart the systemd service. Sudo is scoped via /etc/sudoers.d.
echo "[3/4] restart $SERVICE_NAME"
sudo -n systemctl restart "$SERVICE_NAME"

# 4. Health check. The service takes a moment to bind the port, hence the loop.
echo "[4/4] health check ($HEALTH_URL)"
for ((i = 1; i <= HEALTH_TRIES; i++)); do
  if curl -fsS --max-time 3 "$HEALTH_URL" >/dev/null 2>&1; then
    echo "  ✓ healthy after ${i} attempt(s)"
    echo
    echo "=== Deploy successful ==="
    exit 0
  fi
  echo "  attempt $i/$HEALTH_TRIES — not ready, sleeping ${HEALTH_SLEEP}s"
  sleep "$HEALTH_SLEEP"
done

echo "ERROR: health check failed after $((HEALTH_TRIES * HEALTH_SLEEP))s" >&2
echo "Service status:" >&2
sudo -n systemctl status "$SERVICE_NAME" --no-pager || true
exit 1
