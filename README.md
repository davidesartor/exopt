# exopt

Bayesian optimization of exoskeleton torque profiles. The controller (on the Pi)
estimates the gait phase, applies the current torque profile and streams samples
over ZMQ; the BO loop (on the Mac) proposes a profile, collects a segment of
samples for it, records the trial, and repeats — one continuous experiment.

## Profiles

A torque profile is a periodic function of the gait phase `phi in [0, 2pi)`, as
a fraction of `MAX_TORQUE`: a sparse expansion in periodized RBF atoms sharing
one lengthscale `rho`,

```
f(phi) = sum_{k=1..K} weights_k atom(phi - centers_k; rho)
```

with `K = --atoms`. An atom is the squared exponential wrapped around the cycle,
`sum_m exp(-(t + 2pi m)^2 / 2rho^2)`, scaled to a peak of 1 at its center, so a
weight is the atom's peak torque at any `rho`. By Poisson summation it is the
cosine series `1 + 2 sum_{n>=1} exp(-(rho n)^2 / 2) cos(n t)` up to a constant,
which is how `profiles.py` evaluates it (64 harmonics, constant term included,
so profiles are not zero-mean); its peak, at the center, is the sum of the
coefficients. A profile is stored as its atoms and evaluated through its Fourier
series (`Profile.series`), which the controller computes once per profile.

The acquisition optimizes centers, weights and `rho` jointly within the search
box (`rho` within `RHO_RANGE`, weights within `±MAX_WEIGHT = 0.5`) from the first
trial on, from a screened Latin hypercube. Nothing rescales the summed profile:
the controller clips it at `MAX_TORQUE` (and the drives enforce the same limit).

The surrogate is a GP indexed by functions: squared exponential in the L2
distance between profiles over the gait cycle, computed from their Fourier
coefficients (`profiles.features`, scaled so the euclidean distance is the L2
distance). Profiles with any `rho` or `K` are compared directly. Each trial is a
measurement, its mean loss with a batch-means SE, and the GP takes the SE as known
noise, plus a fitted nugget for what varies between segments beyond it (drift,
fatigue, adaptation):

```
cov(y_i, y_j) = signal^2 k(f_i, f_j) + (nugget^2 + se_i^2) [i = j]
```

The lengthscale, signal variance and nugget are fit by restricted maximum
likelihood, with the constant trend integrated out. Candidates maximize log
expected improvement over the lowest posterior mean among the observed profiles.

`--mode coefficients` is the non-functional baseline: the same loop, but
profiles are a single atom (`--atoms` forced to 1) and the GP is a vanilla
squared exponential on its three coefficients `(center, weight, rho)`, each
mapped from its search range to the unit interval (`profiles.coefficients`). The
center is treated as a plain coordinate, not a periodic one. The GP itself is the
same in both modes, euclidean on an embedding of the profiles
(`profiles.EMBEDDINGS`): Fourier features for `functional`, coefficients for
`coefficients`.

## Running

```bash
# on the Pi (or anywhere with --mock: simulated legs responding to the torque)
uv run exo-controller [--mock]

# on the Mac, through an ssh tunnel to the Pi
ssh -N -L 5555:localhost:5555 -L 5556:localhost:5556 user@pi &
uv run bo-loop --exp-dir <dir> --atoms 2 --duration 30
uv run bo-loop --exp-dir <dir> --mode coefficients   # single atom, vanilla GP

# dashboard: what runs now, how the optimization goes, any trial in detail (or --animation)
uv run --group dev exo-dashboard <dir> [--mode coefficients]   # as bo-loop
```

The dashboard is one page in three bands. **Now** (refreshed at 10 Hz): status tiles (stream, gait
period, profile gain, the current segment's loss estimate), the applied profile against
the live commanded torque, the segment's running estimate against its stop threshold,
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
is calibrated (after "start walking"). Every trial starts from zero torque: the
controller ramps the profile's gain from 0 to 1 over 5 s, and only swaps profiles
at zero gain. Samples count once at full gain, minus the first `--warmup`, until the
segment is unlikely to beat the best: its running loss sits more than `--z`
(default 3) standard deviations above the GP posterior mean of the best observed
profile, `(loss - mu_best) / sqrt(se^2 + sd_best^2) > z`, with `se` by batch
means. Promising segments run to `--max-samples`, as do initial-design segments
(no surrogate yet). Then `bo-loop` releases the profile, which ramps back down to
zero over 5 s while the next proposal is computed. Each trial is recorded as
`<dir>/trial_i.json`, so `bo-loop` resumes an interrupted session. The objective
is minus the mean assistance power, negative power penalized, always minimized;
each trial reports it with its SE, which the GP uses as the measurement noise.

## Layout

```
src/exopt/
  profiles.py           # Profile of periodized RBF atoms, its Fourier series, GP embeddings
  gaussian_process.py   # GP surrogate on embedded profiles, known SE plus a fitted nugget
  acquisition.py        # search box, log-EI, start designs, restarts
  trials.py             # trial_i.json records, the objective and its SE
  link.py               # ZMQ pub/sub link and segment collection
  bo_loop.py            # bo-loop: propose, collect, record, repeat
  controller.py         # exo-controller: phase estimation, torque, drives or simulated legs
  dashboard.py          # exo-dashboard: now / optimization / trial bands, gait animation
docs/
  periodic_kernel.html  # interactive explorer of the periodized kernels
```
