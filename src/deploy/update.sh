#!/bin/bash
# Solar Sentinel — Pi update / deploy script
#
# Pulls the latest code from the current branch's upstream, syncs Python
# dependencies, re-installs picamera2 (which uv sync strips because it's
# not in the lockfile), and restarts the systemd service.
#
# Usage on the Pi:
#     /opt/solar-sentinel/deploy/update.sh
# Or remotely from your laptop:
#     ssh bahapie@<pi> 'sudo /opt/solar-sentinel/deploy/update.sh'
#
# Idempotent: safe to run repeatedly even if there are no new commits.
# Exits non-zero on any step failure.

set -euo pipefail

# Resolve repo root from script location (deploy/ lives at the repo root on the Pi)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE="solar-sentinel"

cd "$REPO_DIR"

echo "── solar-sentinel update ──"
echo "Repo:    $REPO_DIR"
echo "Branch:  $(git rev-parse --abbrev-ref HEAD)"
echo "Before:  $(git rev-parse --short HEAD)"

# 1. Fetch + fast-forward
git fetch --quiet origin
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse '@{u}')
if [[ "$LOCAL" == "$REMOTE" ]]; then
    echo "Already up to date with $REMOTE — skipping git pull."
    NEW_COMMITS=0
else
    git pull --ff-only
    NEW_COMMITS=1
fi
echo "After:   $(git rev-parse --short HEAD)"

# 2. Sync Python deps (always — covers the case where pyproject.toml didn't
#    change but lockfile resolution was incomplete on a previous run).
echo
echo "── uv sync ──"
uv sync

# 3. Re-install picamera2. uv sync strips it because it's not in pyproject.toml
#    (Linux-only build dep python-prctl breaks cross-platform lockfile resolution).
#    Skip if not running on aarch64 Linux (e.g. testing this script on a dev box).
if [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "aarch64" ]]; then
    echo
    echo "── pi-only: picamera2 ──"
    if ! .venv/bin/python -c "import picamera2" 2>/dev/null; then
        uv pip install picamera2
    else
        echo "picamera2 already importable — skipping."
    fi
fi

# 4. Restart the service if it's installed
if systemctl list-unit-files | grep -q "^${SERVICE}.service"; then
    echo
    echo "── restart ${SERVICE} ──"
    sudo systemctl restart "$SERVICE"
    sleep 2
    sudo systemctl status "$SERVICE" --no-pager -l | head -25
else
    echo
    echo "Note: ${SERVICE}.service not installed — skipping restart."
    echo "      First-time deploy: see deploy/README.md."
fi

echo
echo "── done ──"
if [[ "$NEW_COMMITS" == 1 ]]; then
    echo "Updated to $(git rev-parse --short HEAD). Service restarted."
else
    echo "No new commits. Deps re-synced and service restarted."
fi
