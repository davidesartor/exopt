"""Acquisition utilities: stable log-EI, start designs, restart optimization, loss."""

from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import vlse
from jaxtyping import Array, Float, Scalar

from exopt.rkhs_functions import EPS


@jax.jit
def log_expected_improvement(mu: Scalar, sigma: Scalar, y_best: Scalar) -> Scalar:
    """Stable log EI: log(sigma) + log(pdf(z) + z * cdf(z)) (Ament et al. 2023)."""
    z = (y_best - mu) / sigma
    c1 = jnp.log(2 * jnp.pi) / 2
    c2 = jnp.log(jnp.pi / 2) / 2

    # sanitize inputs for the three branches to avoid NaNs and Infs in gradients
    upper, lower = z > -1, z < -1 / jnp.sqrt(EPS)

    # branch1 (z > -1): direct evaluation
    z1 = jnp.where(upper, z, 0.0)
    log_h1 = jnp.log(jsp.stats.norm.pdf(z1) + z1 * jsp.stats.norm.cdf(z1))

    # branch2 (-1/sqrt(EPS) <= z <= -1): stable log1mexp trick
    z2 = jnp.where(upper | lower, -2.0, z)
    log_h2 = (
        -(z2**2) / 2
        - c1
        + jax.nn.log1mexp(-jnp.log(-z2 * jsp.special.erfcx(-z2 / jnp.sqrt(2.0))) - c2)
    )

    # branch3 (z < -1/sqrt(EPS)): asymptotic expansion
    z3 = jnp.where(lower, z, -2.0 / EPS)
    log_h3 = -(z3**2) / 2 - c1 - 2 * jnp.log(-z3)

    log_h = jnp.where(upper, log_h1, jnp.where(lower, log_h3, log_h2))
    return jnp.log(sigma) + log_h


def latin_hypercube(
    dim: int, n: int, seed: int, domain: tuple[float, float] = (0.0, 1.0)
) -> Float[Array, "n d"]:
    rng = np.random.default_rng(seed)
    cells = np.stack([rng.permutation(n) for _ in range(dim)], axis=-1)
    u = (cells + rng.uniform(size=(n, dim))) / n
    return jnp.array(u * (domain[1] - domain[0]) + domain[0])


def optimize_restarts(
    loss: Callable[[Float[Array, "p"]], Scalar],
    candidates: Float[Array, "n p"],
    bounds: tuple[Float[Array, "p"], Float[Array, "p"]],
    max_restarts: int = 5,
) -> Float[Array, "p"]:
    """Screen candidates by loss, polish the best few with L-BFGS-B, keep the winner."""
    screened = jax.jit(jax.vmap(loss))(candidates)
    starts = candidates[jnp.argsort(screened)[:max_restarts]]

    solve = lambda x0: vlse.optim.minimise(loss, x0, bounds=bounds)
    results = jax.jit(jax.vmap(solve))(starts)
    return results.x[jnp.argmin(results.f)]


def loss_function(trace: np.ndarray, beta: float = 5.0) -> float:
    power = np.stack([
        trace["torque_0"] * trace["velocity_0"],
        trace["torque_1"] * trace["velocity_1"],
    ])
    positive = power[power > 0]
    negative = power[power < 0]
    return float((positive.sum() + beta * negative.sum()) / power.size)
