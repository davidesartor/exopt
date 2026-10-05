# exopt

Bayesian optimization of exoskeleton torque profiles. The controller (on the Pi)
estimates the gait phase, applies the current torque profile and streams samples
over ZMQ; the BO loop (on the Mac) proposes a profile, collects a segment of
samples for it, records the trial, and repeats — one continuous experiment.

## Profiles

A torque profile is a truncated Fourier series over the gait phase
`phi in [0, 2pi)`, as a fraction of `MAX_TORQUE`:
`f(phi) = sum_{m=1..H} sin_m * sin(m*phi) + cos_m * cos(m*phi)`, with
`H = --harmonics` (a history with fewer harmonics is zero-padded).

Candidates are searched in a coefficient box shrinking as `m^-2`, then scaled
down so the profile's peak stays within the amplitude cap (ramping from 0.15 to
1 over the first 20 trials). The surrogate is a GP with a squared-exponential
kernel on per-harmonic weighted coefficient distances, weights learned by
maximum likelihood; candidates maximize log expected improvement.

## Running

```bash
# on the Pi (or anywhere with --mock: simulated legs responding to the torque)
uv run exo-controller [--mock]

# on the Mac, through an ssh tunnel to the Pi
ssh -N -L 5555:localhost:5555 -L 5556:localhost:5556 user@pi &
uv run bo-loop --exp-dir <dir> --harmonics 2 --duration 30

# dashboard: live leg state, BO progress, per-trial detail (or --animation)
uv run --group dev exo-dashboard <dir>
```

The controller streams from startup; torque stays zero until the gait frequency
is calibrated (after "start walking"). Each new profile is blended in over 5 s;
samples count once the blend is complete, minus the first `--warmup`, until the
mean converges (`--tol`, capped at `--max-samples`). Each trial is recorded as
`<dir>/trial_i.json`, so `bo-loop` resumes an interrupted session. The objective
is minus the mean assistance power, negative power penalized, always minimized.

## Layout

```
src/exopt/
  profiles.py           # Fourier-coefficient Profile and its spectrum
  gaussian_process.py   # GP surrogate with learned per-harmonic metric
  acquisition.py        # search box, peak limit, log-EI, start designs, restarts
  trials.py             # trial_i.json records and the objective
  link.py               # ZMQ pub/sub link and segment collection
  bo_loop.py            # bo-loop: propose, collect, record, repeat
  controller.py         # exo-controller: phase estimation, torque, drives or simulated legs
  dashboard.py          # exo-dashboard: live view, progress, trial detail, gait animation
```
