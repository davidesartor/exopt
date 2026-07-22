"""Propose the next config to run, from everything observed so far."""

import argparse
import os

import jax

from src import experiment_io, strategy

jax.config.update("jax_enable_x64", True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", required=True, help="experiment folder")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-samples", type=int, default=1024)
    parser.add_argument("--max-restarts", type=int, default=16)
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="minimize the objective (default: maximize)",
    )
    # functional experiments only (detected from the folder, not passed in)
    parser.add_argument(
        "--k", type=int, default=4, help="number of adaptive basis points"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="read/write the experiment folder on this machine instead of the Pi",
    )
    args = parser.parse_args()

    if args.local:
        experiment_io.use_local_storage()
        os.makedirs(args.exp_dir, exist_ok=True)

    def report(n: int, ys):
        if n == 0:
            raise SystemExit(
                f"No completed (config, run) pairs in {args.exp_dir}; "
                "run initial_design.py and mock_experiment.py first."
            )
        print(f"Continuing a {mode} experiment.")
        print(f"Loaded {n} observations, best so far: {ys.min():.6f}")
        print("Fitting surrogate and optimizing expected improvement...")

    # what to optimize is decided by whatever the history already holds
    mode = experiment_io.experiment_mode(args.exp_dir)
    if mode is None:
        raise SystemExit(
            f"No configs in {args.exp_dir}; run initial_design.py first."
        )

    sign = 1.0 if args.minimize else -1.0
    i = experiment_io.next_index(args.exp_dir)

    if mode == "functional":
        fs, ys = experiment_io.load_functional_dataset(args.exp_dir)
        report(len(fs), sign * ys)
        f_next, _ = strategy.propose_next_functional(
            fs,
            sign * ys,
            k=args.k,
            seed=args.seed,
            acquisition_raw_samples=args.raw_samples,
            acquisition_max_restarts=args.max_restarts,
        )
        experiment_io.save_config_function(args.exp_dir, i, f_next)
        print(f"Proposed next profile -> config_{i}.json ({args.k} basis points)")
        print(f"  lengthscale rho: {f_next.rho}")
        print(f"  basis phases:    {f_next.x.squeeze(-1)}")
    else:
        xs, ys = experiment_io.load_dataset(args.exp_dir)
        report(len(xs), sign * ys)
        x_next, _ = strategy.propose_next(
            xs,
            sign * ys,
            seed=args.seed,
            acquisition_raw_samples=args.raw_samples,
            acquisition_max_restarts=args.max_restarts,
        )
        experiment_io.save_config(args.exp_dir, i, x_next)
        print(f"Proposed next point -> config_{i}.txt: {x_next}")


if __name__ == "__main__":
    main()
