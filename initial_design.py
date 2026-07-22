"""Write an initial space-filling design of configs into an experiment folder."""

import argparse
import os

import jax.numpy as jnp

from src import designs, experiment_io, strategy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", required=True, help="experiment folder")
    parser.add_argument(
        "--mode",
        choices=["vector", "functional"],
        default="vector",
        help="parameter vectors, or torque profiles over gait phase",
    )
    parser.add_argument("--dim", type=int, default=1, help="parameter dimension")
    parser.add_argument("--n", type=int, default=2, help="number of configs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--edges",
        action="store_true",
        help="prioritize the edges of the domain instead of Latin hypercube",
    )
    # functional mode only
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="basis points per profile; one support point is the usual seed",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=None,
        help="pin the lengthscale instead of adapting it (fixed-rho baseline)",
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

    kind = "edge-prioritized" if args.edges else "Latin hypercube"
    start = experiment_io.next_index(args.exp_dir)

    if args.mode == "functional":
        print(f"Sampling {args.n} {kind} profiles (k={args.k})...")
        fs = strategy.initial_functions(
            k=args.k,
            n=args.n,
            seed=args.seed,
            edges=args.edges,
            rho=None if args.rho is None else jnp.full(1, args.rho),
        )
        for offset, f in enumerate(fs):
            i = start + offset
            experiment_io.save_config_function(args.exp_dir, i, f)
            print(f"  wrote config_{i}.json: basis phases {f.x.squeeze(-1)}")
    else:
        print(f"Sampling {args.n} {kind} configs (dim={args.dim})...")
        sampler = designs.edge_prioritized if args.edges else designs.latin_hypercube
        xs = sampler(args.dim, args.n, args.seed, domain=designs.VECTOR_DOMAIN)
        for offset, x in enumerate(xs):
            i = start + offset
            experiment_io.save_config(args.exp_dir, i, x)
            print(f"  wrote config_{i}.txt: {x}")

    print(f"Done. {args.n} configs written to {args.exp_dir}")


if __name__ == "__main__":
    main()
