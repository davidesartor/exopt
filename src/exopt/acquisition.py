"""Acquisition: search box and peak limit, stable log-EI, start designs, restart optimization."""

from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import vlse

from jaxtyping import Array, Float, Key, PyTree, Scalar
from exopt import EPS
from exopt.profiles import Profile, spectrum

# profiles are fractions of MAX_TORQUE: the peak may reach the limit,
# but only after the amplitude cap has ramped up over the first trials
AMPLITUDE = 1.0
INITIAL_AMPLITUDE = 0.15
AMPLITUDE_TRIALS = 20
PEAK_GRID = 256  # gait phases on which a profile's peak is evaluated


def amplitude_cap(trials: int) -> float:
    """Amplitude allowed after `trials` observations, ramping to full scale."""
    ramp = min(trials / AMPLITUDE_TRIALS, 1.0)
    return INITIAL_AMPLITUDE + ramp * (AMPLITUDE - INITIAL_AMPLITUDE)


def coefficient_bounds(
    harmonics: int, amplitude: float = AMPLITUDE
) -> tuple[Profile, Profile]:
    """Coefficient box as Profile bounds, shrinking with the decay."""
    scale = amplitude * spectrum(harmonics)
    return Profile(-scale, -scale), Profile(scale, scale)


def box_points(unit: Float[Array, "n D"], lower: Profile, upper: Profile) -> Profile:
    """Map unit-cube points (sin then cos coordinates) onto the box, as a batched Profile."""
    H = lower.harmonics
    u = Profile(unit[:, :H], unit[:, H:])
    return jax.tree.map(lambda lo, hi, uu: lo + uu * (hi - lo), lower, upper, u)


def limit_peak(f: Profile, amplitude: float = AMPLITUDE) -> Profile:
    """Scale a profile down so its peak over the gait cycle is at most `amplitude`."""
    phase = jnp.linspace(0.0, 2 * jnp.pi, PEAK_GRID, endpoint=False)
    peak = jnp.abs(f(phase)).max()
    shrink = jnp.minimum(1.0, amplitude / jnp.maximum(peak, EPS))
    return jax.tree.map(lambda c: shrink * c, f)


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
