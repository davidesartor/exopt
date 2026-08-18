"""Acquisition utilities: stable log-EI, start designs, restart optimization, objective."""

from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np
import vlse

from jaxtyping import Array, Float, Key, PyTree, Scalar
from exopt import EPS
from exopt.rkhs_functions import Profile, spectrum

# largest allowed amplitude of the fundamental harmonic
AMPLITUDE = 1.0


def coefficient_bounds(harmonics: int) -> tuple[Profile, Profile]:
    """Coefficient box as Profile bounds, shrinking with the decay."""
    scale = AMPLITUDE * spectrum(harmonics)
    return Profile(-scale, -scale), Profile(scale, scale)


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


def acquisition_loss(surrogate, y_best: Scalar, f: Profile) -> Scalar:
    """Negative log-EI of a candidate Profile under the surrogate."""
    mu, cov = surrogate.predict(f.pad_to(surrogate.x.harmonics))
    return -log_expected_improvement(
        mu=mu.squeeze(), sigma=cov.squeeze() ** 0.5, y_best=y_best
    )


def latin_hypercube(
    key: Key, dim: int, n: int, domain: tuple[float, float] = (0.0, 1.0)
) -> Float[Array, "n d"]:
    """Space-filling design: one point per row and column of a stratified grid."""
    key_cells, key_jitter = jr.split(key)
    cells = jax.vmap(jr.permutation, in_axes=(0, None), out_axes=1)(
        jr.split(key_cells, dim), n
    )
    u = (cells + jr.uniform(key_jitter, (n, dim))) / n
    return u * (domain[1] - domain[0]) + domain[0]


def optimize_restarts(
    loss: Callable[[PyTree], Scalar],
    candidates: PyTree,
    bounds: tuple[PyTree, PyTree],
    max_restarts: int = 5,
) -> PyTree:
    """Screen candidates by loss, polish the best few with L-BFGS-B, keep the winner."""
    # cheap vectorized screening of all raw candidates (stacked on the leading axis)
    screened = jax.jit(jax.vmap(loss))(candidates)
    best = jnp.argsort(screened)[:max_restarts]
    starts = jax.tree.map(lambda c: c[best], candidates)

    # polish the surviving starts in parallel and keep the best local optimum
    solve = lambda x0: vlse.optim.minimise(loss, x0, bounds=bounds)
    results = jax.jit(jax.vmap(solve))(starts)
    winner = jnp.argmin(results.f)
    return jax.tree.map(lambda a: a[winner], results.x)


def objective(trace: np.ndarray, beta: float = 5.0) -> float:
    """Mean assistance power over a trace, penalizing negative power by beta."""
    power0 = trace["torque_0"] * trace["velocity_0"]
    power1 = trace["torque_1"] * trace["velocity_1"]
    power = jnp.stack([power0, power1])
    power = jnp.where(power < 0, beta * power, power)  # penalize negative power more
    return float(jnp.mean(power))
