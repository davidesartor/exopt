"""Space-filling initial designs over a box domain.

Samplers draw on the unit cube and are mapped onto ``domain`` at the end.
Two domains are in play, and they are not the same thing:

``VECTOR_DOMAIN``
    The search domain of a plain parameter vector -- what a config *is*. Now
    symmetric about zero.

``UNIT``
    The internal encoding of a functional candidate, which ``rkhs.Function``
    decodes into basis locations, values, and a lengthscale, each with its own
    range. That encoding is [0,1] by construction and is not a domain choice.
"""

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
    """Map samples from the unit cube onto ``domain``."""
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
    """Space-filling samples warped toward the domain edges.

    Starts with the literal corners of the domain (its 2^d vertices), in a
    seed-shuffled order so no corner is systematically favored when n < 2^d,
    and the extremes are always evaluated first. Any remaining slots are filled
    from a Latin hypercube pushed toward the faces via the Beta(a, a) inverse
    CDF with a = concentration < 1, which is U-shaped and piles probability
    mass at the boundary.
    """
    corners = jnp.array(list(itertools.product([0.0, 1.0], repeat=dim)))
    order = np.random.default_rng(seed).permutation(corners.shape[0])
    corners = corners[order]
    if n <= corners.shape[0]:
        return rescale(corners[:n], domain)

    n_extra = n - corners.shape[0]
    u = sp.stats.qmc.LatinHypercube(d=dim, rng=seed).random(n=n_extra)
    extra = jnp.array(sp.stats.beta.ppf(u, concentration, concentration))
    return rescale(jnp.concatenate([corners, extra], axis=0), domain)
