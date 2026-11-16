#!/usr/bin/env bash
# One-time Pi setup: its own wifi network "exo-pi" (Pi at 10.42.0.1), up on boot ahead of other wifi.
# Usage: pi-hotspot.sh <wifi password, 8+ chars>. Switch: sudo nmcli con up UMASS / sudo nmcli con up exo-hotspot
set -euo pipefail
password=${1:?usage: pi-hotspot.sh <wifi password>}

sudo nmcli con delete exo-hotspot 2>/dev/null || true
sudo nmcli con add type wifi ifname wlan0 con-name exo-hotspot ssid exo-pi \
  802-11-wireless.mode ap 802-11-wireless.band bg 802-11-wireless.channel 6 \
  wifi-sec.key-mgmt wpa-psk wifi-sec.proto rsn wifi-sec.pairwise ccmp wifi-sec.group ccmp wifi-sec.psk "$password" \
  ipv4.method shared ipv4.addresses 10.42.0.1/24 ipv6.method disabled \
  connection.autoconnect yes connection.autoconnect-priority 10
echo "exo-hotspot saved: up on next boot, or now with: sudo nmcli con up exo-hotspot"
