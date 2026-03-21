#!/bin/bash
# =============================================================================
# Solar Sentinel — First Boot Setup Script
# Runs once on first boot to configure Tailscale and start the application.
#
# Usage:
#   1. Flash Raspberry Pi OS to SD card
#   2. Place your Tailscale auth key in /boot/config/tailscale-authkey.txt
#   3. Copy this script to /opt/solar-sentinel/first-boot.sh
#   4. Enable via: sudo systemctl enable first-boot
#   5. Power on — the Pi will auto-join your Tailscale network
# =============================================================================

set -euo pipefail

LOG="/var/log/solar-sentinel-setup.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== Solar Sentinel first-boot setup — $(date) ==="

# 1. System updates
echo "[1/6] Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# 2. Install Tailscale
echo "[2/6] Installing Tailscale..."
if ! command -v tailscale &> /dev/null; then
    curl -fsSL https://tailscale.com/install.sh | sh
fi

# 3. Authenticate with pre-generated auth key
echo "[3/6] Configuring Tailscale..."
AUTH_KEY_FILE="/boot/config/tailscale-authkey.txt"
if [ -f "$AUTH_KEY_FILE" ]; then
    AUTH_KEY=$(cat "$AUTH_KEY_FILE" | tr -d '[:space:]')
    sudo tailscale up --authkey="$AUTH_KEY" --hostname=solar-sentinel
    echo "Tailscale connected as 'solar-sentinel'"
    # Remove auth key for security
    sudo rm -f "$AUTH_KEY_FILE"
else
    echo "WARNING: No auth key found at $AUTH_KEY_FILE"
    echo "Run 'sudo tailscale up' manually to connect."
fi

# 4. Install Python dependencies
echo "[4/6] Setting up Python environment..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
cd /opt/solar-sentinel
uv sync
uv pip install picamera2

# 5. Enable and start the service
echo "[5/6] Enabling Solar Sentinel service..."
sudo cp deploy/solar-sentinel.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable solar-sentinel
sudo systemctl start solar-sentinel

# 6. Disable first-boot service
echo "[6/6] Disabling first-boot script..."
sudo systemctl disable first-boot

echo "=== Setup complete! ==="
echo "Access the dashboard at: http://solar-sentinel:8000"
echo "Or via Tailscale: http://solar-sentinel.tail<your-tailnet>.ts.net:8000"
