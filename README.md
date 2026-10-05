# exopt

Bayesian optimization of exoskeleton torque profiles. The controller (the Pi in
real trials) streams power samples over ZMQ and swaps torque profiles on the
fly; the BO driver runs locally, collecting a segment of samples per candidate,
fitting the surrogate, and publishing the next profile — one continuous
experiment, many acquisitions.

## Profiles

A torque profile is a truncated Fourier series over the gait phase
`phi in [0, 2pi)`, as a fraction of `MAX_TORQUE`:
`f(phi) = sum_{m=1..H} sin_m * sin(m*phi) + cos_m * cos(m*phi)`.
The phase increases with time and is 0 at peak (centered) leg position.

- **vector** — `H = 1`, a single sinusoid (two coefficients).
- **functional** — `H = --harmonics`; a vector history continues seamlessly,
  zero-padded to the wider space.

Candidates are searched in a coefficient box shrinking as `m^-2`, then scaled
down so the profile's peak stays within the amplitude cap (which ramps from
0.15 to 1 over the first 20 trials), so the controller never clips. The
surrogate is a GP with a squared-exponential kernel on per-harmonic weighted
coefficient distances, weights learned by maximum likelihood.

## Running

```bash
# on the host: the mock controller (or the real one, pdo_controller.py, on the Pi)
uv run mock-stream

# reach a remote host through an ssh tunnel
ssh -N -L 5555:localhost:5555 -L 5556:localhost:5556 user@host &

# locally: seed, then segment-collect / fit / propose / publish, in a loop
uv run bo-loop --exp-dir <dir> --duration 30

# dashboards: BO progress of a folder, and live leg state from the Pi
uv run exo-view <dir>
uv run python -m exopt.live_view dashboard --host localhost
```

Each segment runs one profile until its mean converges (`--tol`, capped at
`--max-samples`). The controller blends from the previous profile over 5 s;
samples count only once the blend is complete, and then the first `--warmup`
are dropped. Segments are recorded to `--exp-dir` as `config_i.json` /
`run_i.txt` pairs, so `bo-loop` resumes an interrupted session. The objective is
minus the mean assistance power (negative power penalized), always minimized;
`bo-loop` keeps proposing until `--duration` minutes elapse.

The controller needs only a non-blocking poll of the profile socket in its main
loop and a `profile_id` (plus `blend`, if it blends) on each streamed sample
(see `zmq_link.controller_link`).

## Layout

```
src/exopt/
  bo_loop.py            # bo-loop: BO proposal and the online loop
  gaussian_process.py   # distance-based GP surrogate
  rkhs_functions.py     # Fourier-coefficient Profile and its geometry
  acquisition.py        # log-EI, start designs, peak limit, restart optimization, loss
  zmq_link.py           # ZMQ pub/sub link, segment collection, payload encoding
  experiment_io.py      # config_i.json / run_i.txt folder records
  mock_controller.py    # mock-stream: simulated gait stand-in for the controller
  pdo_controller.py     # real controller on the Pi: drives the exo over pyCandle
  live_view.py          # encoder zeroing and live position/velocity dashboard
  visualizations.py     # exo-view: BO progress dashboard and gait animation
```
