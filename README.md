# exopt

Bayesian optimization of exoskeleton torque profiles: the controller streams samples over ZMQ, `bo-loop` proposes profiles and records trials.

Double-click `start.command` (macOS), `start.desktop` (linux; first time right-click → Allow Launching) or `start.bat`
(Windows) for the dashboard. It pulls this checkout first, and on
each connect sends the controller host this checkout's commits over ssh (cloning it there if missing; the host needs no internet) before running the controller. Or:

```bash
scripts/pi-hotspot.sh <password>                   # once on the Pi: its own wifi "exo-pi", Pi at 10.42.0.1 (controller "pi (hotspot)")
uv run --group controller src/exopt/controller.py   # on the Pi
uv run --group controller src/exopt/diagnose.py     # on the Pi: drives answer and track a gait torque (or start --diagnose)
uv run src/exopt/mock_controller.py                 # simulated legs
uv run --group optimizer bo-loop --exp-dir <dir> [--atoms 2]
uv run --group optimizer start <dir>                # dashboard
uv run --group optimizer exo-review <dir>           # static <dir>/dashboard.html (--animation: gait animation)
```
