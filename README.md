# exopt

Bayesian optimization driver for the exo experiments. Configs are written to an
experiment folder on the Pi over SFTP, the controller runs them and writes back
a trace, and `loss_function` reduces that trace to a scalar objective.

Two things can be optimized, and the experiment folder decides which:

| mode | a config is | file |
|---|---|---|
| `vector` | a parameter vector in `[-1, 1]^d` | `config_{i}.txt`, one line of floats |
| `functional` | a torque profile over the gait cycle | `config_{i}.json` |

`bo_step.py` infers the mode from the configs already in the folder, so it takes
no `--mode` flag — only `initial_design.py`, which creates the history, does.

## Loop

```bash
# 1. seed the folder
uv run initial_design.py --exp-dir <dir> --dim 4 --n 8
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
`Function` itself, so the history reloads exactly. It also carries a rendered
`samples` lookup table so the controller can read the profile without knowing
anything about the RKHS; the basis is the source of truth.

Ported from the `ours_adaptive` method in the BOFUS repository.

## Mock experiment

`mock_experiment.py` runs the whole loop against a synthetic objective, with no
hardware attached.

```bash
uv run mock_experiment.py --exp-dir <dir> --target Branin --dim 2
uv run mock_experiment.py --exp-dir <dir> --mode functional --target sinc1d --k 4
```

Functional targets: `sinc1d` (BOFUS's benchmark), `ProfileMatch` (recover a
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
`best_guess.png` (functional: best profile against the target curve; vector:
best point on its landscape, one panel per run). Run folders are gitignored;
the summary CSVs are kept.

## Layout

```
bo_step.py           # propose the next config
initial_design.py    # seed a folder with a space-filling design
mock_experiment.py   # run the loop end to end against a synthetic target
summarize.py         # aggregate run folders into a CSV
plots.py             # convergence and best-guess figures
plot_surrogate.py    # surrogate mean over a 2D slice
src/
  strategy.py        # propose_next (vector) and propose_next_functional
  gp.py              # GaussianProcess and FunctionalGaussianProcess
  rkhs.py            # Function with adaptive rho, ambient RKHS
  acquisition.py     # log-EI, UCB, multi-restart optimization
  designs.py         # Latin hypercube and edge-prioritized designs
  kernels.py         # metrics and kernel profiles
  targets.py         # functional test objectives
  virtual_library.py # scalar benchmark functions
  experiment_io.py   # SFTP config/run IO and the objective
```
