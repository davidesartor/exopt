# exopt

Bayesian optimization driver for the exo experiments. Configs are written to an
experiment folder on the Pi over SFTP, the controller runs them and writes back
a trace, and `loss_function` reduces that trace to a scalar objective.

Both modes optimize the same thing — a torque profile over the gait cycle — and
write the same kind of config. What differs is how much of the curve the search
may move:

| mode | a config is | free parameters |
|---|---|---|
| `vector` | one sine over the cycle | amplitude, phase shift (2) |
| `functional` | a free-form RKHS profile | basis points, coefficients, `rho` |

Every config is a `config_{i}.json` carrying a `"mode"` key, so a folder says
which mode to continue in and the controller renders either one. `vector` is the
traditional-BO baseline the functional method is meant to beat: same objective,
same folder format, same plots, fewer degrees of freedom in the curve.

`bo_step.py` infers the mode from the configs already in the folder, so it takes
no `--mode` flag — only `initial_design.py`, which creates the history, does.

## Loop

```bash
# 1. seed the folder
uv run initial_design.py --exp-dir <dir> --n 8
uv run initial_design.py --exp-dir <dir> --mode functional --n 8

# 2. run the configs on the hardware, then propose the next one
uv run bo_step.py --exp-dir <dir>
```

`bo_step.py` maximizes by default; pass `--minimize` to flip it. Every script
talks to the Pi over SFTP; pass `--local` to work on this machine instead
(`mock_experiment.py` is local by default and takes `--remote` to opt in).

## Functional mode

A profile is a sparse RKHS function: `k` basis points, their coefficients, and
its **own lengthscale `rho`**. The acquisition optimizes all three jointly, so
how smooth the profile should be is chosen by the search instead of fixed up
front. `rho` is searched in log scale over `rkhs.RHO_RANGE`. Pass `--rho` to pin
it instead — the fixed-lengthscale baseline this method is meant to improve on.

The initial design seeds with a single support point (`--k 1`, the default);
proposals use `--k` basis points. Basis size may differ between observations,
so the two need not match.

Candidates with different `rho` are not elements of one space, so they are
compared in an *ambient* squared-exponential RKHS sitting at the bottom of that
range, which has a closed-form inner product (`rkhs.RKHS.inner`). This is also
why `k` may differ between observations: the metric never sees the
parameterization, and basis padding is exact.

`config_{i}.json` stores `rho`, the basis points, and the coefficients — the
`Function` itself, so the history reloads exactly. The controller evaluates the
basis expansion to get the curve; there is no rendered copy that can drift.

Ported from the `ours_adaptive` method in the BOFUS repository.

## The controller

`PDO_Walking_profile.py` runs on the Pi. It reads the newest `config_{i}.json`,
renders whichever mode it holds, and logs the trace to `run_{i}.txt`.

The profile's input is the **gait phase**, estimated per leg from that leg's own
phase-plane angle, `atan2(velocity * dt, position)`, shifted from `(-pi, pi]` to
`[0, 1)`. Each leg therefore gets the torque the curve asks for at the point in
the stride that leg is actually at, and the two run about half a cycle apart.
Position and velocity are low-passed before the angle is taken — the phase wraps
at 0/1, and filtering a wrapping signal drags it across the whole cycle on every
wrap.

## Vector mode

A config is `{"mode": "vector", "amplitude": a, "phase": p}`, both in `[-1, 1]`,
and the curve is `a * sin(2*pi*(t - (p + 1)/2))` over gait phase `t`. The GP and
log-EI machinery is unchanged — it just searches a 2-D cube. The amplitude is a
real parameter here, so the controller applies the sine as it is, where an RKHS
profile is peak-normalized first (its coefficients carry no scale of their own).

## Mock experiment

`mock_experiment.py` runs the whole loop against a synthetic objective, with no
hardware attached.

```bash
uv run mock_experiment.py --exp-dir <dir> --target ProfileMatch
uv run mock_experiment.py --exp-dir <dir> --mode functional --target sinc1d --k 4
```

Both modes take the same targets, which is what makes the two comparable:
`sinc1d` (BOFUS's benchmark), `ProfileMatch` (recover a
reference torque curve), or any `virtual_library` name, lifted to the functional
domain via `targets.Ridge`. Note `Ridge` at `d=1` probes the candidate at a
single point, so it collapses to a 1-D problem — use `sinc1d` or `ProfileMatch`
to exercise the functional machinery properly.

## Results

Both tools read a folder of run folders, taking the label from the folder name
and splitting a trailing `_s{seed}`:

```bash
uv run summarize.py --root results/sweep --out results/summary.csv
uv run plots.py --root results/sweep --out results/plots --target sinc1d
```

`plots.py` writes `convergence.png` (best-so-far per evaluation, log y) and
`best_guess.png` (the best profile each label found, against the target curve).
Sine and adaptive runs can sit in one sweep folder and are drawn on the same
axes. Vector runs also get `best_params.png`, the amplitude/phase plane they
searched. Run folders are gitignored; the summary CSVs are kept.

## Layout

```
bo_step.py           # propose the next config
initial_design.py    # seed a folder with a space-filling design
mock_experiment.py   # run the loop end to end against a synthetic target
summarize.py         # aggregate run folders into a CSV
plots.py             # convergence and best-guess figures
plot_surrogate.py    # surrogate mean over a 2D slice
PDO_Walking_profile.py  # runs on the Pi: renders either config, logs the trace
src/
  strategy.py        # propose_next (vector) and propose_next_functional
  sine.py            # the two-parameter sine profile vector mode searches
  gp.py              # GaussianProcess and FunctionalGaussianProcess
  rkhs.py            # Function with adaptive rho, ambient RKHS
  acquisition.py     # log-EI, UCB, multi-restart optimization
  designs.py         # Latin hypercube and edge-prioritized designs
  kernels.py         # metrics and kernel profiles
  targets.py         # functional test objectives
  virtual_library.py # scalar benchmark functions
  experiment_io.py   # SFTP config/run IO and the objective
```
