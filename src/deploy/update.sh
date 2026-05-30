#!/bin/bash
# Solar Sentinel — Pi update / deploy script
#
# Pulls the latest code from the current branch's upstream, syncs Python
# dependencies, re-installs picamera2 (which uv sync strips because it's
# not in the lockfile), and restarts the systemd service.
#
# Usage on the Pi:
#     /opt/solar-sentinel/src/deploy/update.sh
# Or via the convenience symlink (created in deploy/README.md):
#     sudo solar-sentinel-update
# Or remotely from your laptop:
#     ssh bahapie@<pi> 'sudo solar-sentinel-update'
#
# Idempotent: safe to run repeatedly even if there are no new commits.
# Exits non-zero on any step failure.

set -euo pipefail

# Resolve script location, dereferencing symlinks (e.g. /usr/local/bin/solar-sentinel-update).
# `readlink -f` walks every symlink and returns the canonical path of the script itself.
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# The git working tree may live anywhere up the directory tree from the script.
# In the canonical Solar Sentinel layout, .git is at /opt/solar-sentinel/ but
# this script lives at /opt/solar-sentinel/src/deploy/update.sh. Walk up until
# we find the .git dir.
REPO_DIR="$SCRIPT_DIR"
while [[ "$REPO_DIR" != "/" && ! -d "$REPO_DIR/.git" ]]; do
    REPO_DIR="$(dirname "$REPO_DIR")"
done
if [[ ! -d "$REPO_DIR/.git" ]]; then
    echo "ERROR: could not find .git/ above $SCRIPT_DIR" >&2
    exit 1
fi

# The Python project (pyproject.toml + venv) lives at <repo>/src on this layout.
# If pyproject.toml is at the repo root instead (flat layout), use the repo root.
if [[ -f "$REPO_DIR/src/pyproject.toml" ]]; then
    APP_DIR="$REPO_DIR/src"
elif [[ -f "$REPO_DIR/pyproject.toml" ]]; then
    APP_DIR="$REPO_DIR"
else
    echo "ERROR: no pyproject.toml found at $REPO_DIR or $REPO_DIR/src" >&2
    exit 1
fi

SERVICE="solar-sentinel"

# Resolve uv binary. sudo strips PATH down to /etc/sudoers' secure_path, so
# uv installed at $HOME/.local/bin/uv is invisible. Try common locations.
UV_BIN=""
for candidate in \
    "$(command -v uv 2>/dev/null || true)" \
    "/home/$(logname 2>/dev/null || echo "$SUDO_USER")/.local/bin/uv" \
    "$HOME/.local/bin/uv" \
    "/usr/local/bin/uv" \
    "/usr/bin/uv"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        UV_BIN="$candidate"
        break
    fi
done
if [[ -z "$UV_BIN" ]]; then
    echo "ERROR: could not locate the 'uv' binary." >&2
    echo "       Install with:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    echo "       Or set UV_BIN explicitly:  UV_BIN=/path/to/uv solar-sentinel-update" >&2
    exit 1
fi
# Allow caller to override
UV_BIN="${UV_BIN_OVERRIDE:-$UV_BIN}"

cd "$REPO_DIR"

echo "── solar-sentinel update ──"
echo "Repo:    $REPO_DIR"
echo "App:     $APP_DIR"
echo "uv:      $UV_BIN"
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
cd "$APP_DIR"
"$UV_BIN" sync

# 3. Re-install picamera2. uv sync strips it because it's not in pyproject.toml
#    (Linux-only build dep python-prctl breaks cross-platform lockfile resolution).
#    Skip if not running on aarch64 Linux (e.g. testing this script on a dev box).
if [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "aarch64" ]]; then
    echo
    echo "── pi-only: picamera2 ──"
    if ! .venv/bin/python -c "import picamera2" 2>/dev/null; then
        "$UV_BIN" pip install picamera2
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

