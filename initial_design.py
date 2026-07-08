import argparse
import itertools

import jax.numpy as jnp
import numpy as np
import scipy as sp
from jaxtyping import Array, Float

from src import experiment_io


def latin_hypercube(dim: int, n: int, seed: int) -> Float[Array, "n d"]:
    sampler = sp.stats.qmc.LatinHypercube(d=dim, rng=seed)
    return jnp.array(sampler.random(n=n))


def edge_prioritized(
    dim: int, n: int, seed: int, concentration: float = 0.5
) -> Float[Array, "n d"]:
    """Space-filling samples warped toward the domain edges.

    Starts with the literal corners of the domain (the 2^d vertices of the
    unit cube), in a seed-shuffled order so no corner is systematically
    favored when n < 2^d, and the extremes are always evaluated first. Any
    remaining slots are filled from a Latin hypercube pushed toward 0 or 1
    via the Beta(a, a) inverse CDF with a = concentration < 1, which is
    U-shaped and piles probability mass at the boundary.
    """
    corners = jnp.array(list(itertools.product([0.0, 1.0], repeat=dim)))
    order = np.random.default_rng(seed).permutation(corners.shape[0])
    corners = corners[order]
    if n <= corners.shape[0]:
        return corners[:n]

    n_extra = n - corners.shape[0]
    u = sp.stats.qmc.LatinHypercube(d=dim, rng=seed).random(n=n_extra)
    extra = jnp.array(sp.stats.beta.ppf(u, concentration, concentration))
    return jnp.concatenate([corners, extra], axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", required=True, help="experiment folder")
    parser.add_argument("--dim", type=int, required=True, help="parameter dimension")
    parser.add_argument("--n", type=int, default=2, help="number of configs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--edges",
        action="store_true",
        help="prioritize the edges of the domain instead of Latin hypercube",
    )
    args = parser.parse_args()

    if args.edges:
        xs = edge_prioritized(args.dim, args.n, args.seed)
        print(f"Sampling {args.n} edge-prioritized configs (dim={args.dim})...")
    else:
        xs = latin_hypercube(args.dim, args.n, args.seed)
        print(f"Sampling {args.n} Latin hypercube configs (dim={args.dim})...")

    start = experiment_io.next_index(args.exp_dir)
    for offset, x in enumerate(xs):
        i = start + offset
        experiment_io.save_config(args.exp_dir, i, x)
        print(f"  wrote config_{i}.txt: {x}")

    print(f"Done. {args.n} configs written to {args.exp_dir}")


if __name__ == "__main__":
    main()
