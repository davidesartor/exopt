"""Plot a set of experiment folders: loss over time, and the final best guess.

    uv run plots.py --root results/sweep --out results/plots --target sinc1d

Produces two figures:

``convergence.png``
    Best objective so far against evaluation number, one line per run label.
    Log y, because the losses span decades and a linear axis hides everything
    that happens after the first few evaluations.

``best_guess.png``
    What the search actually landed on. For a functional experiment that is the
    best profile drawn against the target curve it was trying to reproduce; for
    a vector experiment it is the best point on the objective landscape.
"""

import argparse
import os
import re

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from src import designs, experiment_io, targets, virtual_library

# Fixed-rho baselines are an ordered quantity, so they take an ordinal ramp
# (one hue, light->dark by lengthscale) rather than unrelated categorical hues:
# the reader can see the ordering without consulting the legend. The adaptive
# run is a different kind of thing, so it takes a contrasting categorical hue.
# Validated with the dataviz palette checks: ordinal ramp passes step-lightness
# (dL >= 0.09) and nearest-surface contrast (2.06:1); adaptive separates from
# every ramp step at normal-vision dE >= 27.8 and CVD dE >= 23.1.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#104281"]
# categorical order for non-ramp series, taken in fixed order and never cycled
CATEGORICAL = ["#eb6834", "#2a78d6", "#1baf7a"]
ADAPTIVE = CATEGORICAL[0]
# the target is a reference, not a competitor: near-zero chroma so it cannot be
# mistaken for a series, but heavy enough to read underneath them
TARGET_INK = "#b0afa9"
INK, MUTED, GRID = "#1a1a19", "#5c5c57", "#e6e5e1"


def place_end_labels(ax, items, min_gap=0.055, log=False):
    """Direct-label lines at the right edge, pushed apart so they never overlap.

    ``items`` is [(y_value, text, color)]. Positions are solved in axes
    fractions: sort by height, then walk upward enforcing a minimum gap, and if
    that overflows the top, walk back down. A faint leader connects each label
    to the height it actually belongs to.
    """
    lo, hi = ax.get_ylim()
    to_frac = (lambda v: (np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))) if log \
        else (lambda v: (v - lo) / (hi - lo))

    order = sorted(range(len(items)), key=lambda i: items[i][0])
    placed = {}
    prev = -1e9
    for i in order:
        f = max(to_frac(items[i][0]), prev + min_gap)
        placed[i] = f
        prev = f
    overflow = max(placed.values()) - 1.0
    if overflow > 0:  # ran off the top: shift the whole stack down
        for i in placed:
            placed[i] -= overflow

    for i, (value, text, color) in enumerate(items):
        true_f, label_f = to_frac(value), placed[i]
        if abs(true_f - label_f) > 0.005:  # only draw a leader if it was moved
            ax.annotate(
                "",
                xy=(1.0, true_f),
                xytext=(1.035, label_f),
                xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.5),
            )
        ax.annotate(
            text,
            xy=(1.045, label_f),
            xycoords="axes fraction",
            va="center",
            fontsize=9,
            color=color,
        )


def load_runs(root: str) -> dict[str, list]:
    """Every run folder under root, grouped by label with _s{seed} split off."""
    runs = {}
    for name in sorted(os.listdir(root)):
        exp_dir = os.path.join(root, name)
        if not os.path.isdir(exp_dir):
            continue
        mode = experiment_io.experiment_mode(exp_dir)
        if mode is None:
            continue
        m = re.match(r"(.+)_s(\d+)$", name)
        label, seed = (m.group(1), int(m.group(2))) if m else (name, -1)
        if mode == "functional":
            xs, ys = experiment_io.load_functional_dataset(exp_dir)
        else:
            xs, ys = experiment_io.load_dataset(exp_dir)
        runs.setdefault(label, []).append(dict(seed=seed, mode=mode, xs=xs, ys=ys))
    return runs


def series_color(label: str, ordered_labels: list[str]) -> str:
    """Ordinal ramp for the fixed-rho family, categorical hues for everything else."""
    fixed = [l for l in ordered_labels if l.startswith("fixed")]
    if label in fixed:
        # spread the family across the ramp, darkest = largest lengthscale
        i = fixed.index(label)
        return RAMP[round(i * (len(RAMP) - 1) / max(len(fixed) - 1, 1))]
    # remaining labels take the categorical order, assigned in fixed order
    others = [l for l in ordered_labels if l not in fixed]
    return CATEGORICAL[others.index(label) % len(CATEGORICAL)]


def plot_convergence(runs: dict, out: str) -> str:
    """Best-so-far against evaluation count, median across seeds with a spread band."""
    labels = sorted(runs, key=lambda l: (not l.startswith("adaptive"), l))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    fig.subplots_adjust(left=0.10, right=0.78, top=0.90, bottom=0.11)
    end_labels = []
    for label in labels:
        color = series_color(label, labels)
        curves = [np.minimum.accumulate(np.asarray(r["ys"])) for r in runs[label]]
        n = min(len(c) for c in curves)
        stack = np.stack([c[:n] for c in curves])
        steps = np.arange(1, n + 1)
        median = np.median(stack, axis=0)

        if len(stack) > 1:  # spread only means something with more than one seed
            ax.fill_between(
                steps,
                np.quantile(stack, 0.25, axis=0),
                np.quantile(stack, 0.75, axis=0),
                color=color,
                alpha=0.15,
                lw=0,
            )
        ax.plot(steps, median, color=color, lw=2, solid_capstyle="round")
        end_labels.append((float(median[-1]), label, color))

    seeds = {len(v) for v in runs.values()}
    n_seed = f"{min(seeds)}" if len(seeds) == 1 else f"{min(seeds)}-{max(seeds)}"
    ax.set_yscale("log")
    ax.set_xlabel("evaluation")
    ax.set_ylabel("best objective so far")
    ax.set_title(
        f"Convergence  ({n_seed} seed{'s' if n_seed != '1' else ''} per configuration)",
        color=INK,
        loc="left",
    )
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    # direct labels, so identity never rests on color alone
    place_end_labels(ax, end_labels, log=True)

    path = os.path.join(out, "convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_best_functional(runs: dict, target_fn, out: str) -> str:
    """The best profile found, drawn against the curve it was trying to match."""
    grid = experiment_io.profile_grid(400)
    labels = sorted(runs, key=lambda l: (not l.startswith("adaptive"), l))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    fig.subplots_adjust(left=0.10, right=0.74, top=0.90, bottom=0.11)
    ax.plot(
        grid.squeeze(-1),
        target_fn.reference(grid),
        color=TARGET_INK,
        lw=4.5,
        zorder=1,
        label="target",
    )

    best_overall = None
    end_labels = []
    for label in labels:
        color = series_color(label, labels)
        run = min(runs[label], key=lambda r: float(jnp.min(r["ys"])))
        j = int(jnp.argmin(run["ys"]))
        f, loss = run["xs"][j], float(run["ys"][j])
        curve = f.sample(grid)
        ax.plot(grid.squeeze(-1), curve, color=color, lw=2, zorder=2)
        end_labels.append((float(curve[-1]), f"{label}  {loss:.4f}", color))
        if best_overall is None or loss < best_overall[1]:
            best_overall = (f, loss, color)

    # mark where the winner put its basis points: the adaptive part, made visible
    f, loss, color = best_overall
    ax.scatter(
        f.x.squeeze(-1),
        f.sample(f.x),
        s=70,
        color=color,
        ec="white",
        lw=1.5,
        zorder=4,
        label=f"basis points (best run, rho={float(f.rho[0]):.3f})",
    )

    ax.set_xlabel("gait phase")
    ax.set_ylabel("torque")
    ax.set_title("Best profile found vs target", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    place_end_labels(ax, end_labels, min_gap=0.07)

    path = os.path.join(out, "best_guess.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_best_vector(runs: dict, target_fn, out: str, seed: int = 0) -> str:
    """The best point found, on the landscape it was searching.

    One panel per label: different labels are usually different objectives, and
    drawing them over a single contour would put one run's points on another
    run's landscape. Each panel resolves its own landscape from its label, and
    falls back to the ``--target`` one when the label does not name a benchmark.
    """
    labels = sorted(runs)
    dim = int(next(iter(runs.values()))[0]["xs"].shape[-1])

    if dim != 2:
        # no landscape to draw above 2D: show the winning point coordinate by coordinate
        fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
        for label in labels:
            color = series_color(label, labels)
            run = min(runs[label], key=lambda r: float(jnp.min(r["ys"])))
            j = int(jnp.argmin(run["ys"]))
            ax.plot(
                range(1, dim + 1),
                np.asarray(run["xs"][j]),
                marker="o",
                ms=7,
                lw=2,
                color=color,
                label=f"{label} best {float(run['ys'][j]):.4g}",
            )
        ax.axhline(0.0, color=MUTED, lw=1, ls=":", zorder=0)
        ax.set_xlabel("coordinate")
        ax.set_ylabel("value")
        ax.set_ylim(*designs.VECTOR_DOMAIN)
        ax.set_title(f"Best point found  ({dim}D)", color=INK, loc="left")
        ax.legend(frameon=False, fontsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    else:
        n = len(labels)
        fig, axes = plt.subplots(
            1, n, figsize=(4.6 * n, 4.4), constrained_layout=True, squeeze=False
        )
        for ax, label in zip(axes[0], labels):
            # the panel title carries identity here, so the marks are free to use
            # the one hue that stays legible on the blue landscape
            color = ADAPTIVE
            run = min(runs[label], key=lambda r: float(jnp.min(r["ys"])))
            xs, j = run["xs"], int(jnp.argmin(run["ys"]))

            # each panel gets its own landscape, resolved from its own label
            panel_target = build_target(label, "vector", seed) or target_fn
            if panel_target is not None:
                g = jnp.linspace(*designs.VECTOR_DOMAIN, 200)
                gx, gy = jnp.meshgrid(g, g)
                z = panel_target(jnp.stack([gx, gy], axis=-1))
                cs = ax.contourf(gx, gy, z, levels=30, cmap="Blues_r")
                fig.colorbar(cs, ax=ax, label="objective", shrink=0.85)

            ax.scatter(
                xs[:, 0], xs[:, 1], s=26, color=color, alpha=0.5, ec="none", zorder=3
            )
            ax.scatter(
                xs[j, 0],
                xs[j, 1],
                s=300,
                marker="*",
                color=color,
                ec="white",
                lw=1.5,
                zorder=4,
            )
            ax.annotate(
                f"best {float(run['ys'][j]):.3g}",
                (float(xs[j, 0]), float(xs[j, 1])),
                textcoords="offset points",
                xytext=(10, 10),
                fontsize=9,
                color="white",
                zorder=5,
            )
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_title(f"{label}  ({len(run['ys'])} evaluations)", color=INK, loc="left")

    path = os.path.join(out, "best_guess.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_target(name: str, mode: str, seed: int = 0):
    """Rebuild the objective a sweep was run against, for the reference curve.

    Returns None when the name does not resolve to a benchmark, so a caller can
    fall back rather than crash on a label that is not a target name.
    """
    if name is None:
        return None
    if mode == "vector" and not hasattr(virtual_library, name):
        return None
    if mode == "functional":
        if name in ("SincProjection", "sinc1d"):
            return targets.SincProjection(d=1, seed=seed)
        if name == "ProfileMatch":
            return targets.ProfileMatch()
        return targets.Ridge(getattr(virtual_library, name)(), d=1, seed=seed)
    profile = getattr(virtual_library, name)()
    lo, hi = designs.VECTOR_DOMAIN
    return lambda x: profile((x - lo) / (hi - lo))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="folder holding run folders")
    parser.add_argument("--out", default="results/plots", help="where to write figures")
    parser.add_argument("--target", help="objective the runs used, for the reference")
    parser.add_argument("--seed", type=int, default=1, help="seed the target was built with")
    parser.add_argument(
        "--remote", action="store_true", help="read run folders from the Pi"
    )
    args = parser.parse_args()

    if not args.remote:
        experiment_io.use_local_storage()
    os.makedirs(args.out, exist_ok=True)

    runs = load_runs(args.root)
    if not runs:
        raise SystemExit(f"No run folders found under {args.root}")
    mode = next(iter(runs.values()))[0]["mode"]

    print(f"Loaded {sum(len(v) for v in runs.values())} runs "
          f"({len(runs)} labels, {mode} mode)")
    print("Wrote", plot_convergence(runs, args.out))

    target_fn = build_target(args.target, mode, args.seed)
    if mode == "functional":
        if target_fn is None:
            print("Skipping best_guess.png: pass --target to draw the reference curve")
            return
        print("Wrote", plot_best_functional(runs, target_fn, args.out))
    else:
        print("Wrote", plot_best_vector(runs, target_fn, args.out, args.seed))


if __name__ == "__main__":
    main()
