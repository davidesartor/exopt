import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jaxtyping import Array, Float, Scalar

import bo_step
import initial_design
from src import experiment_io, gp, virtual_library

jax.config.update("jax_enable_x64", True)

# blue target/landscape, orange evaluations, green best-so-far; the best point
# also carries a star marker + label so identity never rests on color alone
LINE, EVALS, BEST, INK = "#4C72B0", "#DD8452", "#55A868", "#333333"


def plot_ei(
    exp_dir: str,
    iteration: int,
    surrogate_model: gp.GaussianProcess,
    x_next: Float[Array, "2"],
    resolution: int = 150,
) -> str:
    """Save an expected-improvement heatmap over the 2D domain for one iteration."""
    observed_xs = surrogate_model.observed_xs
    y_best = surrogate_model.observed_ys.min()

    # plot log-EI: raw EI spans orders of magnitude and collapses to ~0 near
    # observed points, so a linear heatmap washes out; log is what's optimized
    @jax.jit
    def log_ei_at(x: Float[Array, "2"]) -> Scalar:
        mu, cov = surrogate_model.predict(x[None, :])
        return gp.log_expected_improvement(
            mu=mu.squeeze(), sigma=cov.squeeze() ** 0.5, y_best=y_best
        )

    g = jnp.linspace(0, 1, resolution)
    gx, gy = jnp.meshgrid(g, g)
    pts = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)
    log_ei = np.exp(jax.vmap(log_ei_at)(pts).reshape(gx.shape))

    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    hm = ax.imshow(
        log_ei, origin="lower", extent=(0, 1, 0, 1), aspect="auto", cmap="viridis"
    )
    fig.colorbar(hm, ax=ax, label="log expected improvement")
    ax.scatter(
        observed_xs[:, 0],
        observed_xs[:, 1],
        c="white",
        ec="black",
        s=45,
        zorder=3,
        label="observed",
    )
    ax.scatter(
        x_next[0],
        x_next[1],
        c="red",
        marker="*",
        s=350,
        ec="white",
        zorder=4,
        label="next (EI max)",
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"iteration {iteration}: log expected improvement", color=INK)
    ax.legend(frameon=True, loc="upper right", fontsize=8)

    path = os.path.join(exp_dir, f"bo_step_{iteration}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_experiment(exp_dir, target_fn, target_name):
    """Save a landscape (1D/2D) or convergence (higher-D) plot of all evals."""
    xs, ys = experiment_io.load_dataset(exp_dir)
    dim = xs.shape[-1]
    best = int(jnp.argmin(ys))

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    if dim == 1:
        grid = jnp.linspace(0, 1, 400)[:, None]
        ax.plot(grid[:, 0], target_fn(grid), color=LINE, lw=2, label=target_name)
        ax.scatter(xs[:, 0], ys, color=EVALS, s=40, zorder=3, label="evaluations")
        ax.scatter(xs[best, 0], ys[best], color=BEST, marker="*", s=320, zorder=4)
        ax.annotate(
            f"best {ys[best]:.3f}",
            (xs[best, 0], ys[best]),
            textcoords="offset points",
            xytext=(8, 8),
            color=INK,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("objective")
        ax.legend(frameon=False)

    elif dim == 2:
        g = jnp.linspace(0, 1, 200)
        gx, gy = jnp.meshgrid(g, g)
        z = target_fn(jnp.stack([gx, gy], axis=-1))
        cs = ax.contourf(gx, gy, z, levels=30, cmap="Blues_r")
        fig.colorbar(cs, ax=ax, label="objective")
        ax.scatter(xs[:, 0], xs[:, 1], color=EVALS, s=40, zorder=3, ec="white")
        ax.scatter(
            xs[best, 0],
            xs[best, 1],
            color=BEST,
            marker="*",
            s=320,
            zorder=4,
            ec="white",
        )
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    else:
        running_best = jnp.minimum.accumulate(ys)
        steps = range(1, len(ys) + 1)
        ax.plot(steps, running_best, color=LINE, lw=2, marker="o", ms=4)
        ax.set_xlabel("evaluation")
        ax.set_ylabel("best objective so far")

    ax.set_title(f"{target_name}  ({len(ys)} evaluations, {dim}D)", color=INK)
    ax.grid(True, color="0.9", zorder=0)

    path = os.path.join(exp_dir, "mock_experiment.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def mock_run(y: float, timesteps: int = 100) -> pd.DataFrame:
    """Fabricate a hardware-like run dataframe whose loss_function recovers y.

    Stands in for the exo controller's mechanicalPower trace. The virtual_library
    targets are shifted so y >= 0, so a constant positive power series has
    loss_function == mean(power) == y exactly.
    """
    power = np.full(timesteps, y)
    return pd.DataFrame(
        {
            "time": np.linspace(0, 1, timesteps),
            "mechanicalPower_0": power,
            "mechanicalPower_1": power,
        }
    )


def evaluate_pending(exp_dir, target_fn):
    """Mock-evaluate every config that has no run file yet."""
    pending = sorted(
        experiment_io.config_numbers(exp_dir) - experiment_io.run_numbers(exp_dir)
    )
    for i in pending:
        x = experiment_io.read_config(exp_dir, i)
        experiment_io.save_run(exp_dir, i, mock_run(float(target_fn(x))))
    return pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", required=True, help="experiment folder")
    parser.add_argument("--target", default="Ackley", help="virtual_library function")
    parser.add_argument("--dim", type=int, default=2, help="parameter dimension")
    parser.add_argument("--initial", type=int, default=6, help="initial design size")
    parser.add_argument("--iterations", type=int, default=12, help="BO iterations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--edges", action="store_true", help="edge-prioritized initial design"
    )
    args = parser.parse_args()

    target_fn = getattr(virtual_library, args.target)()

    # step 1: initial design
    if args.edges:
        xs = initial_design.edge_prioritized(args.dim, args.initial, args.seed)
    else:
        xs = initial_design.latin_hypercube(args.dim, args.initial, args.seed)
    for offset, x in enumerate(xs):
        experiment_io.save_config(args.exp_dir, offset + 1, x)

    # step 2: evaluate the initial design
    evaluate_pending(args.exp_dir, target_fn)
    xs, ys = experiment_io.load_dataset(args.exp_dir)
    print(f"Initial design: {len(xs)} points, best so far: {ys.min():.6f}")

    # step 3: BO loop -- propose, plot EI heatmap, evaluate
    for it in range(args.iterations):
        xs, ys = experiment_io.load_dataset(args.exp_dir)
        x_next, surrogate_model = bo_step.propose_next(xs, ys, seed=args.seed + it)

        i = experiment_io.next_index(args.exp_dir)
        experiment_io.save_config(args.exp_dir, i, x_next)
        if args.dim == 2:
            plot_ei(args.exp_dir, i, surrogate_model, x_next)

        y_next = float(target_fn(x_next))
        experiment_io.save_run(args.exp_dir, i, mock_run(y_next))
        print(
            f"Iteration {it + 1}/{args.iterations}: "
            f"x={x_next}, y={y_next:.6f}, best={min(ys.min(), y_next):.6f}"
        )

    # final: heatmap of the true target function with every evaluation
    path = plot_experiment(args.exp_dir, target_fn, args.target)
    xs, ys = experiment_io.load_dataset(args.exp_dir)
    print(f"\nDone. Best found: {ys.min():.6f}")
    print(f"Saved final function heatmap to {path}")


if __name__ == "__main__":
    main()
