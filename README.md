# exopt

Bayesian optimization of exoskeleton torque profiles. The controller (on the Pi)
estimates the gait phase, applies the current torque profile and streams samples
over ZMQ; the BO loop (on the Mac) proposes a profile, collects a segment of
samples for it, records the trial, and repeats — one continuous experiment.

## Profiles

A torque profile is a zero-mean periodic function of the gait phase
`phi in [0, 2pi)`, as a fraction of `MAX_TORQUE`: a sparse expansion in
periodized RBF atoms sharing one lengthscale `rho`,

```
f(phi) = sum_{k=1..K} weights_k k_per(phi - centers_k; rho)
```

with `K = --atoms`. Any stationary kernel `k` is wrapped around the cycle,
`k_per(t) = sum_m k(t + 2pi m)`; by Poisson summation its Fourier coefficients
are `k`'s spectral density sampled at the harmonics, `lambda_n = S(n) / pi`,
which is how `kernels.py` evaluates it (truncated at 64 harmonics, constant term
dropped). Kernels are given by their log spectral density: squared exponential
(the default) and Matern-nu are provided.

The acquisition optimizes centers, weights and `rho` jointly within the search
box (`rho` within `RHO_RANGE`), from a screened Latin hypercube; candidates are
then scaled down so the peak stays within the amplitude cap (ramping from 0.15
to 1 over the first 20 trials).

The surrogate is a GP indexed by functions: squared exponential in the L2
distance between profiles over the gait cycle, `||f - g||^2 = pi sum_n
|coefficient differences|^2`, computed from Fourier coefficients
(`profiles.features`). For SE atoms the atom inner product is also closed form,
`sqrt(2pi) rho1 rho2 / rho_s k_per(c1 - c2; rho_s)` with
`rho_s^2 = rho1^2 + rho2^2`. Profiles with any `rho` or `K` are compared
directly, with no ambient space to fit; only the GP lengthscale and the nugget
are fit by maximum likelihood. Candidates maximize log expected improvement.

## Running

```bash
# on the Pi (or anywhere with --mock: simulated legs responding to the torque)
uv run exo-controller [--mock]

# on the Mac, through an ssh tunnel to the Pi
ssh -N -L 5555:localhost:5555 -L 5556:localhost:5556 user@pi &
uv run bo-loop --exp-dir <dir> --atoms 2 --duration 30

# dashboard: what runs now, how the optimization goes, any trial in detail (or --animation)
uv run --group dev exo-dashboard <dir>
```

The dashboard is one page in three bands. **Now**: status tiles (stream, gait
period, blend, the current segment's loss estimate), the applied profile against
the live commanded torque, the segment's running estimate with its stopping band,
and live leg traces. **Optimization**: loss per trial (±2 SE by batch means) with
the best so far, a held-out check of the GP surrogate (each trial predicted
without it), every profile tried, and the trials in profile space (principal
coordinates of the L2 distance). **Trial**: click a trial in any of those charts
(or step through them) for its atoms against the measured torque, penalized power
over the gait cycle per leg, the convergence of its estimate, and its phase
portrait. The page follows the OS light/dark setting, and the header button
toggles it. `bo-loop` writes the profile it is running to `<dir>/current.json`.

`docs/periodic_kernel.html` is a standalone explorer of the periodized kernels:
the copies being wrapped around the cycle, the truncated Fourier series against
the exact sum, the spectrum sampled at the harmonics, and two atoms with their
L2 inner product and distance, with sliders for the kernel family, lengthscales,
centers, weights and truncation.

The controller streams from startup; torque stays zero until the gait frequency
is calibrated (after "start walking"). Each new profile is blended in over 5 s;
samples count once the blend is complete, minus the first `--warmup`, until the
mean converges (`--tol`, capped at `--max-samples`). Each trial is recorded as
`<dir>/trial_i.json`, so `bo-loop` resumes an interrupted session. The objective
is minus the mean assistance power, negative power penalized, always minimized.

## Layout

```
src/exopt/
  kernels.py            # periodized stationary kernels via their spectral densities
  profiles.py           # Profile of periodized RBF atoms, L2 distance
  gaussian_process.py   # GP surrogate on the L2 distance
  acquisition.py        # search box, peak limit, log-EI, start designs, restarts
  trials.py             # trial_i.json records and the objective
  link.py               # ZMQ pub/sub link and segment collection
  bo_loop.py            # bo-loop: propose, collect, record, repeat
  controller.py         # exo-controller: phase estimation, torque, drives or simulated legs
  dashboard.py          # exo-dashboard: now / optimization / trial bands, gait animation
docs/
  periodic_kernel.html  # interactive explorer of the periodized kernels
```
