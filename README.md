# exopt

Bayesian optimization of exoskeleton torque profiles: the controller on the Pi streams samples over ZMQ, `bo-loop` proposes profiles and records trials.

Double-click `start.command` (macOS), `start.desktop` (linux; first time right-click → Allow Launching) or `start.bat` (Windows) for the dashboard, then Connect. It reaches the Pi over ssh by whichever of these answers:

| route | this computer | Pi address |
| --- | --- | --- |
| wifi | on the lab wifi with the Pi | `exo-pi.local` (or its address, under Settings) |
| hotspot | joined to the Pi's own wifi `exo-pi` (it comes up when the lab wifi is out of range) | `10.42.0.1` |
| usb | USB-C cable to the Pi's USB-C port | `10.55.0.1` |

The “Pi wifi” button in the top bar moves the Pi between the lab wifi and its hotspot (over usb without losing the connection). On each connect the dashboard copies this computer's `src/`, `pyproject.toml` and `uv.lock` to the Pi as they are (committed or not; no git or internet involved) and runs the controller there.

```bash
scripts/pi-setup.sh <hotspot password>              # once on the Pi: hostname, hotspot, usb ethernet
uv run --group optimizer start [<dir>]              # dashboard (--mock: simulated legs here, --diagnose: drive check)
uv run --group optimizer bo-loop --exp-dir <dir>    # optimization alone, against a running controller
uv run --group optimizer exo-review <dir>           # static <dir>/dashboard.html (--animation: gait animation)
uv run --group controller src/exopt/control_loop.py # on the Pi, by hand (also diagnose.py)
```

| module | side | what |
| --- | --- | --- |
| `control_loop`, `gait_estimation`, `drives_md80` | Pi | control loop, gait phase estimator, MD80 drives |
| `diagnose`, `drives_mock` | Pi | drive check, control loop on simulated legs |
| `streams` | both | the ZMQ sample and profile streams |
| `raspberrypi` | laptop | ssh routes and tunnel to the Pi, this computer's code copied to it |
| `optimization_loop`, `experiment` | laptop | optimization and held runs, settings and trial files |
| `dashboard`, `figures`, `review`, `web/` | laptop | live page, plots, static review page |
