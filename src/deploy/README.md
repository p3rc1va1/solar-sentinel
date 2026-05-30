# Deploy

This directory contains everything needed to deploy Solar Sentinel from
GitHub Actions to the Raspberry Pi. Push to `main` → tests run on a free
GitHub-hosted Ubuntu runner → if green, the Pi pulls the new code, syncs
deps, restarts the service, and health-checks itself.

## Files in this directory

| File | Purpose |
|------|---------|
| `solar-sentinel.service` | Systemd unit running the FastAPI app on boot. |
| `first-boot.sh` | One-time provisioning for a freshly flashed SD card (Tailscale, uv, service install). |
| `deploy.sh` | Called by the GitHub Actions deploy job. Rsyncs code, runs `uv sync`, restarts service, health-checks. |
| `update.sh` | One-shot manual update: `git pull` + `uv sync` + reinstall `picamera2` + restart service. Use when not running the GitHub Actions runner. |
| `sudoers-solar-sentinel` | Sudoers fragment letting the runner restart the service without a password. |
| `README.md` | This file. |

The CI/CD entry point is `.github/workflows/deploy.yml` at the repo root.

---

# Pi system prerequisites (do these once)

Before any deploy method works, the Pi needs a few system packages and a
correctly-shaped venv. Skipping any of these will produce confusing errors
that surface later (most often "Camera running in stub mode" even with the
camera plugged in, or a `226/NAMESPACE` failure from systemd).

```bash
# Camera tooling + system Python bindings for libcamera
sudo apt update
sudo apt install -y rpicam-apps python3-picamera2 python3-libcamera libcap-dev

# Verify libcamera sees the camera before going further
rpicam-hello --list-cameras       # must list imx708_wide or similar

# Recreate the venv with --system-site-packages so picamera2 can import
# libcamera (which is a system C++ library, not a pip package)
cd /opt/solar-sentinel
rm -rf .venv
uv venv --system-site-packages
uv sync
uv pip install picamera2          # not in pyproject.toml on purpose; see note below

# Confirm the camera stack is reachable from inside the venv
.venv/bin/python -c "from picamera2 import Picamera2; Picamera2().close(); print('OK')"
```

If `rpicam-hello --list-cameras` shows no cameras, the issue is upstream of
deployment — check the ribbon cable orientation and `/boot/firmware/config.txt`
contains `camera_auto_detect=1`.

**Why `picamera2` isn't in `pyproject.toml`.** Its transitive dep
`python-prctl` is Linux-only and breaks lockfile resolution on macOS dev
machines. Both `update.sh` and `deploy.sh` reinstall it after every `uv sync`.

---

# Day-to-day updates without GitHub Actions

If you don't want to set up a self-hosted runner (the steps below), you have
two simpler options:

**Option A — One command from the Pi:**

```bash
ssh <user>@<pi-host>
sudo /opt/solar-sentinel/deploy/update.sh
```

`update.sh` does: `git pull` (fast-forward only), `uv sync`, reinstall
`picamera2` if missing, `sudo systemctl restart solar-sentinel`, print status.

**Option B — One command from your laptop (after pushing to `main`):**

```bash
ssh <user>@<pi-host> 'sudo /opt/solar-sentinel/deploy/update.sh'
```

Mark the script executable on the Pi the first time:

```bash
ssh <user>@<pi-host> 'chmod +x /opt/solar-sentinel/deploy/update.sh'
```

`update.sh` is idempotent — running it with no new commits just re-syncs deps
and restarts the service.

If you want CI/CD-style deploys triggered automatically on every `main` push,
keep reading the GitHub Actions setup below.

---

# Step-by-step: deploy from scratch

This guide assumes you have:

- A Raspberry Pi 5 with Raspberry Pi OS (64-bit) flashed and reachable via SSH.
- The repo's `first-boot.sh` already run, OR `/opt/solar-sentinel` set up
  manually with the systemd service running.
- A GitHub repo containing this code (e.g. `github.com/<you>/solar-sentinel`).

If `/opt/solar-sentinel` does not exist yet, do **Step 0** first; otherwise
skip to Step 1.

---

## Step 0 — (Optional) Bootstrap the project on the Pi

Skip this if `first-boot.sh` already ran or `/opt/solar-sentinel` already
contains a running install.

```bash
# SSH in (over Tailscale or LAN)
ssh <user>@<pi-host>

# Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Clone and stage the project at the canonical location
sudo mkdir -p /opt/solar-sentinel
sudo chown $USER:$USER /opt/solar-sentinel
git clone https://github.com/<owner>/solar-sentinel.git /tmp/ss
cp -r /tmp/ss/src/* /opt/solar-sentinel/
cd /opt/solar-sentinel
uv sync
uv pip install picamera2          # Pi-only; not in pyproject.toml on purpose

# Install + start the systemd service
sudo cp /tmp/ss/src/deploy/solar-sentinel.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now solar-sentinel

# Sanity check
sudo systemctl status solar-sentinel       # active (running)
curl http://localhost:8000/health          # returns JSON
```

---

## Step 1 — Get a runner registration token from GitHub

1. In your browser, open the repo on GitHub.
2. Go to **Settings → Actions → Runners → New self-hosted runner**.
3. Choose **Linux** as OS and **ARM64** as architecture (the Pi 5 is ARM64 —
   x64 binaries will fail with `Exec format error`).
4. Copy the token from the `./config.sh ... --token <TOKEN>` line shown on
   that page. The token is single-use and expires in ~1 hour.

> **Don't paste this token into chats, issues, or commits.** If you do, get a
> fresh one from the same page — the old one is invalidated automatically.

---

## Step 2 — Install the GitHub Actions runner agent on the Pi

SSH into the Pi as your normal user (e.g. `bahapie`). Pick the latest version
from <https://github.com/actions/runner/releases> — substitute below.

```bash
mkdir -p ~/actions-runner && cd ~/actions-runner

VERSION=2.328.0    # ← replace with current latest

curl -O -L https://github.com/actions/runner/releases/download/v${VERSION}/actions-runner-linux-arm64-${VERSION}.tar.gz
tar xzf ./actions-runner-linux-arm64-${VERSION}.tar.gz

./config.sh \
  --url https://github.com/<owner>/solar-sentinel \
  --token <TOKEN_FROM_STEP_1>
```

`config.sh` will ask a few questions — press **Enter** for every default:

- Runner group → `Default`
- Runner name → defaults to the Pi's hostname
- Labels → `self-hosted, Linux, ARM64`
- Work folder → `_work`

You should see `√ Settings Saved.` at the end.

---

## Step 3 — Register the runner as a systemd service

This makes the runner auto-start on every boot. Use **your username**, not
literally `pi`:

```bash
cd ~/actions-runner
sudo ./svc.sh install $USER
sudo ./svc.sh start
sudo ./svc.sh status        # should show: active (running)
```

Verify on GitHub: **Settings → Actions → Runners** — your runner appears with
a green **Idle** dot. That's the success signal.

---

## Step 4 — Make the deploy target writable by the runner

`deploy.sh` rsyncs into `/opt/solar-sentinel`. The runner runs jobs as your
user, so that user needs write access to the directory:

```bash
sudo chown -R $USER:$USER /opt/solar-sentinel
```

---

## Step 5 — Install the sudoers rule

The runner needs to restart the systemd service without a password prompt.
The fragment in `sudoers-solar-sentinel` grants exactly two commands —
nothing else.

If your runner user is **not** `bahapie`, edit the file first:

```bash
# On the Pi, after Step 0 (so /opt/solar-sentinel/deploy/ exists):
sudo cp /opt/solar-sentinel/deploy/sudoers-solar-sentinel \
        /etc/sudoers.d/solar-sentinel-deploy
sudo chmod 0440 /etc/sudoers.d/solar-sentinel-deploy
sudo visudo -c                    # MUST print "parsed OK"
```

Test it — the next command must run **without prompting for a password**:

```bash
sudo -n systemctl status solar-sentinel
```

If you get `sudo: a password is required`, the rule isn't taking effect. Open
the file with `sudo visudo -f /etc/sudoers.d/solar-sentinel-deploy` and check
the username matches `whoami`.

---

## Step 6 — Verify uv is on the runner's PATH

The runner inherits a minimal environment when run as a service. Confirm `uv`
is reachable:

```bash
sudo -u $USER bash -lc 'which uv'
```

If this prints nothing, add uv's location to the runner's environment. The
simplest fix:

```bash
echo 'PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
sudo systemctl restart "actions.runner.<owner>-solar-sentinel.$(hostname).service"
```

(Use `sudo systemctl list-units 'actions.runner.*'` to find the exact service
name if you're unsure.)

---

## Step 7 — Push to `main` from your dev machine

On your laptop:

```bash
cd /path/to/solar-sentinel
chmod +x src/deploy/deploy.sh        # so git tracks the executable bit
git add .github/ src/deploy/
git commit -m "ci: add github actions deploy to raspberry pi"
git push
```

Open the **Actions** tab on GitHub. You should see two jobs run in sequence:

1. **Test (GitHub-hosted)** — `pytest` + `ruff` on Ubuntu. ~30–60s.
2. **Deploy (self-hosted Pi)** — runs on your runner. rsyncs code,
   `uv sync`, restarts the service, polls `http://localhost:8000/health`.

A green check on both = your Pi is now running the new code.

---

## Step 8 — Confirm on the Pi

```bash
ssh <user>@<pi-host>
sudo systemctl status solar-sentinel       # should show "active (running)"
journalctl -u solar-sentinel -n 50         # recent logs
curl http://localhost:8000/health          # 200 OK
```

You can also watch the runner log live during a deploy:

```bash
journalctl -u 'actions.runner.*' -f
```

---

# Day-to-day: what you do now

Just push to `main`. Every push:

1. Tests run on Ubuntu (free).
2. If they pass, the deploy job runs on your Pi.
3. The Pi rsyncs code (preserving `data/` — DB, captures, model file),
   runs `uv sync --inexact` (preserving the manually-installed `picamera2`),
   restarts the service, and confirms `/health` returns 200.
4. Workflow goes green ↔ Pi has the new code.

If anything fails, the existing service keeps running the previous code. The
workflow goes red on GitHub so you see it immediately.

---

# Troubleshooting

### `Exec format error` when running `config.sh`

You downloaded the x64 runner. The Pi 5 is ARM64. Re-download the
`linux-arm64` tarball (not `linux-x64`).

### Runner shows as "Offline" on GitHub

```bash
sudo systemctl list-units 'actions.runner.*'
sudo systemctl status <the-service-name>
sudo systemctl restart <the-service-name>
```

The runner reconnects automatically when network is available.

### Deploy job fails at `uv sync`

The runner can't find `uv`. See Step 6.

### Deploy job fails at `sudo -n systemctl restart`

The sudoers rule isn't right. Re-check Step 5 — the username in
`/etc/sudoers.d/solar-sentinel-deploy` must match the runner user.

### Health check fails after restart

The service started but isn't healthy. SSH in and check:

```bash
sudo systemctl status solar-sentinel
journalctl -u solar-sentinel -n 100 --no-pager
```

Common causes: missing env vars (e.g. `GEMINI_API_KEY` not in `/opt/solar-sentinel/.env`),
broken Python imports, model file missing.

### picamera2 disappears after a deploy

`uv sync` was run without `--inexact` somewhere. Reinstall:

```bash
cd /opt/solar-sentinel
uv pip install picamera2
sudo systemctl restart solar-sentinel
```

`deploy.sh` uses `--inexact` precisely to avoid this; if you ran a plain
`uv sync` manually, that's what pruned it.

---

# Manual rollback

There's no automated rollback. If a bad commit lands on the Pi:

**Fast way (revert via Git, redeploy):**

```bash
# On your laptop
git revert <bad-sha>
git push
```

The next workflow run brings the Pi back to the reverted state.

**Faster way (SSH and check out the previous commit directly):**

```bash
ssh <user>@<pi-host>
cd /opt/solar-sentinel
# Note: /opt/solar-sentinel is NOT a git checkout — it's an rsync target.
# To roll back you have to re-rsync from a known-good source. Easiest is:
git -C /tmp/ss-rollback clone https://github.com/<owner>/solar-sentinel.git \
  || git -C /tmp/ss-rollback pull
cd /tmp/ss-rollback
git checkout <good-sha>
GITHUB_WORKSPACE=$(pwd) bash src/deploy/deploy.sh
```

---

# Local testing of `deploy.sh`

`deploy.sh` is plain bash and works outside CI. Run it on the Pi against any
checkout to dry-test changes:

```bash
cd ~/some-checkout-of-solar-sentinel
GITHUB_WORKSPACE=$(pwd) bash src/deploy/deploy.sh
```

Override defaults via env vars:

```bash
DEPLOY_TARGET=/tmp/ss-test \
HEALTH_URL=http://localhost:8001/health \
  bash src/deploy/deploy.sh
```
