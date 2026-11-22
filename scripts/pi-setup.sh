#!/usr/bin/env bash
# One-time Pi setup of the three ways the dashboard reaches it (src/exopt/raspberrypi.py ROUTES). Reboot after.
#   wifi     hostname exo-pi (exo-pi.local) on the lab wifi, joined first when in range
#   hotspot  its own wifi "exo-pi" (Pi at 10.42.0.1), up on boot when the lab wifi is not in range
#   usb      ethernet over the USB-C port (Pi at 10.55.0.1), which then also powers the Pi
# Usage: pi-setup.sh <hotspot password, 8+ chars>
set -euo pipefail
password=${1:?usage: pi-setup.sh <hotspot password>}

sudo raspi-config nonint do_hostname exo-pi

# below the lab wifi's default priority 0, so it is the fallback
sudo nmcli con delete exo-hotspot 2>/dev/null || true
sudo nmcli con add type wifi ifname wlan0 con-name exo-hotspot ssid exo-pi \
  802-11-wireless.mode ap 802-11-wireless.band bg 802-11-wireless.channel 6 \
  wifi-sec.key-mgmt wpa-psk wifi-sec.proto rsn wifi-sec.pairwise ccmp wifi-sec.group ccmp wifi-sec.psk "$password" \
  ipv4.method shared ipv4.addresses 10.42.0.1/24 ipv6.method disabled \
  connection.autoconnect yes connection.autoconnect-priority -10

# the USB-C port as an ethernet gadget
grep -q '^dtoverlay=dwc2' /boot/firmware/config.txt || printf '[all]\ndtoverlay=dwc2\n' | sudo tee -a /boot/firmware/config.txt >/dev/null
grep -q 'modules-load=dwc2,g_ether' /boot/firmware/cmdline.txt || sudo sed -i 's/rootwait/rootwait modules-load=dwc2,g_ether/' /boot/firmware/cmdline.txt
# NetworkManager leaves gadgets unmanaged by default
sed 's/\(DEVTYPE}=="gadget".*NM_UNMANAGED}=\)"1"/\1"0"/' /usr/lib/udev/rules.d/85-nm-unmanaged.rules \
  | sudo tee /etc/udev/rules.d/85-nm-unmanaged.rules >/dev/null
sudo nmcli con delete exo-usb 2>/dev/null || true
sudo nmcli con add type ethernet ifname usb0 con-name exo-usb \
  ipv4.method shared ipv4.addresses 10.55.0.1/24 ipv6.method disabled connection.autoconnect yes

echo "done: sudo reboot. Wifi by hand: sudo nmcli con up exo-hotspot / sudo nmcli con down exo-hotspot (back to the lab wifi)"
