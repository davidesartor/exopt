
import itertools

import jax.numpy as jnp
import numpy as np
import scipy as sp
from jaxtyping import Array, Float

UNIT = (0.0, 1.0)
VECTOR_DOMAIN = (-1.0, 1.0)


def rescale(
    u: Float[Array, "n d"], domain: tuple[float, float]
) -> Float[Array, "n d"]:
    return u * (domain[1] - domain[0]) + domain[0]


def latin_hypercube(
    dim: int, n: int, seed: int, domain: tuple[float, float] = UNIT
) -> Float[Array, "n d"]:
    sampler = sp.stats.qmc.LatinHypercube(d=dim, rng=seed)
    return rescale(jnp.array(sampler.random(n=n)), domain)


def edge_prioritized(
    dim: int,
    n: int,
    seed: int,
    concentration: float = 0.5,
    domain: tuple[float, float] = UNIT,
) -> Float[Array, "n d"]:
    corners = jnp.array(list(itertools.product([0.0, 1.0], repeat=dim)))
    order = np.random.default_rng(seed).permutation(corners.shape[0])
    corners = corners[order]
    if n <= corners.shape[0]:
        return rescale(corners[:n], domain)

    n_extra = n - corners.shape[0]
    u = sp.stats.qmc.LatinHypercube(d=dim, rng=seed).random(n=n_extra)
    extra = jnp.array(sp.stats.beta.ppf(u, concentration, concentration))
    return rescale(jnp.concatenate([corners, extra], axis=0), domain)
