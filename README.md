# exopt

Bayesian optimization of exoskeleton torque profiles: the controller (Pi) streams samples over ZMQ, `bo-loop` (Mac)
proposes profiles and records trials.

Double-click `start.command` (macOS) / `start.bat` (Windows) for the dashboard. Or:

```bash
uv run src/exopt/controller.py                      # on the Pi
uv run src/exopt/mock_controller.py                 # simulated legs
uv run bo-loop --exp-dir <dir> [--atoms 2] [--mode coefficients]
uv run --group dev start <dir>                      # dashboard
uv run exo-review <dir>                             # static <dir>/dashboard.html
```
