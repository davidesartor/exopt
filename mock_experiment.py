import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jaxtyping import Array, Float, Scalar

from src import (
    acquisition,
    designs,
    experiment_io,
    gp,
    rkhs,
    strategy,
    targets,
    virtual_library,
)

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
        return acquisition.log_expected_improvement(
            mu=mu.squeeze(), sigma=cov.squeeze() ** 0.5, y_best=y_best
        )

    lo, hi = designs.VECTOR_DOMAIN
    g = jnp.linspace(lo, hi, resolution)
    gx, gy = jnp.meshgrid(g, g)
    pts = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)
    log_ei = np.exp(jax.vmap(log_ei_at)(pts).reshape(gx.shape))

    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    hm = ax.imshow(
        log_ei, origin="lower", extent=(lo, hi, lo, hi), aspect="auto", cmap="viridis"
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
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
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
        grid = jnp.linspace(*designs.VECTOR_DOMAIN, 400)[:, None]
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
        g = jnp.linspace(*designs.VECTOR_DOMAIN, 200)
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


def plot_profiles(exp_dir, fs, ys, target_name):
    """Save every evaluated torque profile, shaded by objective, best on top."""
    grid = experiment_io.profile_grid(200)
    best = int(jnp.argmin(ys))
    order = np.argsort(-np.asarray(ys))  # worst first, so the good ones sit on top

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    cmap = plt.get_cmap("viridis_r")
    norm = plt.Normalize(float(ys.min()), float(ys.max()))
    for j in order:
        ax.plot(
            grid.squeeze(-1), fs[j].sample(grid), color=cmap(norm(ys[j])), lw=1.2, alpha=0.8
        )
    ax.plot(
        grid.squeeze(-1),
        fs[best].sample(grid),
        color=BEST,
        lw=3,
        zorder=5,
        label=f"best {ys[best]:.3f}",
    )
    ax.scatter(
        fs[best].x.squeeze(-1),
        fs[best].sample(fs[best].x),
        color=BEST,
        marker="*",
        s=250,
        ec="white",
        zorder=6,
        label="adaptive basis points",
    )
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="objective")
    ax.set_xlabel("gait phase")
    ax.set_ylabel("torque")
    ax.set_title(f"{target_name}  ({len(ys)} profiles evaluated)", color=INK)
    ax.legend(frameon=False)
    ax.grid(True, color="0.9", zorder=0)

    path = os.path.join(exp_dir, "mock_experiment_profiles.png")
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
    if experiment_io.experiment_mode(exp_dir) == "functional":
        read = experiment_io.read_config_function
    else:
        read = experiment_io.read_config
    for i in pending:
        x = read(exp_dir, i)
        experiment_io.save_run(exp_dir, i, mock_run(float(target_fn(x))))
    return pending


def run_vector(args, target_fn):
    # step 1: initial design
    sampler = designs.edge_prioritized if args.edges else designs.latin_hypercube
    xs = sampler(args.dim, args.initial, args.seed, domain=designs.VECTOR_DOMAIN)
    for offset, x in enumerate(xs):
        experiment_io.save_config(args.exp_dir, offset + 1, x)

    # step 2: evaluate the initial design
    evaluate_pending(args.exp_dir, target_fn)
    xs, ys = experiment_io.load_dataset(args.exp_dir)
    print(f"Initial design: {len(xs)} points, best so far: {ys.min():.6f}")

    # step 3: BO loop -- propose, plot EI heatmap, evaluate
    for it in range(args.iterations):
        xs, ys = experiment_io.load_dataset(args.exp_dir)
        x_next, surrogate_model = strategy.propose_next(xs, ys, seed=args.seed + it)

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


def run_functional(args, target_fn):
    """Same loop, but the search variable is a torque profile, not a vector."""
    rho = None if args.rho is None else jnp.full(1, args.rho)

    # step 1: initial design in function space
    fs = strategy.initial_functions(
        k=args.initial_k, n=args.initial, seed=args.seed, edges=args.edges, rho=rho
    )
    for offset, f in enumerate(fs):
        experiment_io.save_config_function(args.exp_dir, offset + 1, f)

    # step 2: evaluate the initial design
    evaluate_pending(args.exp_dir, target_fn)
    fs, ys = experiment_io.load_functional_dataset(args.exp_dir)
    print(f"Initial design: {len(fs)} profiles, best so far: {ys.min():.6f}")

    # step 3: BO loop -- propose a profile, evaluate it
    for it in range(args.iterations):
        fs, ys = experiment_io.load_functional_dataset(args.exp_dir)
        f_next, _ = strategy.propose_next_functional(
            fs,
            ys,
            k=args.k,
            seed=args.seed + it,
            rho=rho,
        )

        i = experiment_io.next_index(args.exp_dir)
        experiment_io.save_config_function(args.exp_dir, i, f_next)

        y_next = float(target_fn(f_next))
        experiment_io.save_run(args.exp_dir, i, mock_run(y_next))
        print(
            f"Iteration {it + 1}/{args.iterations}: "
            f"y={y_next:.6f}, best={min(ys.min(), y_next):.6f}, "
            f"rho={float(f_next.rho[0]):.3f}, "
            f"basis phases={np.round(f_next.x.squeeze(-1), 3)}"
        )

    fs, ys = experiment_io.load_functional_dataset(args.exp_dir)
    path = plot_profiles(args.exp_dir, fs, ys, args.target)
    print(f"\nDone. Best found: {ys.min():.6f}")
    print(f"Saved profile plot to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", required=True, help="experiment folder")
    parser.add_argument(
        "--mode",
        choices=["vector", "functional"],
        default="vector",
        help="optimize a parameter vector, or a torque profile over gait phase",
    )
    parser.add_argument("--target", default="Ackley", help="virtual_library function")
    parser.add_argument("--dim", type=int, default=2, help="parameter dimension")
    parser.add_argument("--initial", type=int, default=6, help="initial design size")
    parser.add_argument("--iterations", type=int, default=12, help="BO iterations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--edges", action="store_true", help="edge-prioritized initial design"
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="write the experiment folder to the Pi instead of this machine",
    )
    # functional mode only
    parser.add_argument(
        "--k", type=int, default=4, help="number of adaptive basis points"
    )
    parser.add_argument(
        "--initial-k",
        type=int,
        default=1,
        help="basis points per initial-design profile (BOFUS seeds with one)",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=None,
        help="pin the lengthscale instead of adapting it (fixed-rho baseline)",
    )
    args = parser.parse_args()

    # no hardware in the loop here, so stay on this machine unless asked
    if not args.remote:
        experiment_io.use_local_storage()
        os.makedirs(args.exp_dir, exist_ok=True)

    if args.mode == "functional":
        if args.target in ("SincProjection", "sinc1d"):
            # BOFUS's functional benchmark: depends on f everywhere
            target_fn = targets.SincProjection(d=1, seed=args.seed)
        elif args.target == "ProfileMatch":
            # recover a reference torque profile: also depends on f everywhere
            target_fn = targets.ProfileMatch()
        else:
            # lift a scalar benchmark to the functional domain over gait phase
            profile = getattr(virtual_library, args.target)()
            target_fn = targets.Ridge(profile, d=1, seed=args.seed)
        run_functional(args, target_fn)
    else:
        # virtual_library functions are defined on the unit cube, so map the
        # search domain onto it rather than moving every benchmark
        profile = getattr(virtual_library, args.target)()
        lo, hi = designs.VECTOR_DOMAIN
        run_vector(args, lambda x: profile((x - lo) / (hi - lo)))


if __name__ == "__main__":
    main()
