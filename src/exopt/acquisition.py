from typing import Callable
from jaxtyping import Array, Float, Key, PyTree, Scalar

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree
import vlse

from exopt import EPS
from exopt.gaussian_process import GaussianProcess
from exopt.profiles import RHO_RANGE, Profile

# profiles are fractions of MAX_TORQUE: the peak may reach the limit,
# but only after the amplitude cap has ramped up over the first trials
AMPLITUDE = 1.0
INITIAL_AMPLITUDE = 0.15
AMPLITUDE_TRIALS = 20
PEAK_GRID = 256  # gait phases on which a profile's peak is evaluated


################################################################################
# region Search Space


def amplitude_cap(trials: int) -> float:
    """Amplitude allowed after `trials` observations, ramping to full scale."""
    ramp = min(trials / AMPLITUDE_TRIALS, 1.0)
    return INITIAL_AMPLITUDE + ramp * (AMPLITUDE - INITIAL_AMPLITUDE)


def bounds(atoms: int, amplitude: float = AMPLITUDE) -> tuple[Profile, Profile]:
    """Atoms anywhere on the cycle, weights up to the amplitude, rho within RHO_RANGE."""
    lower = Profile(jnp.zeros(atoms), jnp.full(atoms, -amplitude), jnp.asarray(RHO_RANGE[0]))
    upper = Profile(jnp.full(atoms, 2 * jnp.pi), jnp.full(atoms, amplitude), jnp.asarray(RHO_RANGE[1]))
    return lower, upper


def limit_peak(f: Profile, amplitude: float = AMPLITUDE) -> Profile:
    """Scale a profile down so its peak over the gait cycle is at most `amplitude`."""
    peak = jnp.abs(f(jnp.linspace(0.0, 2 * jnp.pi, PEAK_GRID, endpoint=False))).max()
    return f.scale(jnp.minimum(1.0, amplitude / jnp.maximum(peak, EPS)))


def latin_hypercube(key: Key, lower: PyTree, upper: PyTree, n: int) -> PyTree:
    """Space-filling design in the box, stacked on a leading axis."""
    lo, unravel = ravel_pytree(lower)
    hi, _ = ravel_pytree(upper)

    # one point per row and column of a stratified grid
    key_cells, key_jitter = jr.split(key)
    cells = jax.vmap(jr.permutation, (0, None), 1)(jr.split(key_cells, len(lo)), n)
    u = (cells + jr.uniform(key_jitter, (n, len(lo)))) / n
    return jax.vmap(unravel)(lo + u * (hi - lo))


# endregion
################################################################################


################################################################################
# region Expected Improvement


def log_expected_improvement(mu: Scalar, sigma: Scalar, y_best: Scalar) -> Scalar:
    # numerically stable version following https://arxiv.org/pdf/2310.20708
    z = (y_best - mu) / sigma
    c1 = jnp.log(2 * jnp.pi) / 2
    c2 = jnp.log(jnp.pi / 2) / 2

    # sanitize the inputs of every branch, so no NaN or Inf reaches the gradients
    upper, lower = z > -1, z < -1 / jnp.sqrt(EPS)

    # z > -1: direct evaluation
    z1 = jnp.where(upper, z, 0.0)
    log_h1 = jnp.log(jsp.stats.norm.pdf(z1) + z1 * jsp.stats.norm.cdf(z1))

    # -1/sqrt(EPS) <= z <= -1: log1mexp trick
    z2 = jnp.where(upper | lower, -2.0, z)
    log_h2 = -(z2**2) / 2 - c1 + jax.nn.log1mexp(-jnp.log(-z2 * jsp.special.erfcx(-z2 / jnp.sqrt(2.0))) - c2)

    # z < -1/sqrt(EPS): asymptotic expansion
    z3 = jnp.where(lower, z, -2.0 / EPS)
    log_h3 = -(z3**2) / 2 - c1 - 2 * jnp.log(-z3)

    return jnp.log(sigma) + jnp.where(upper, log_h1, jnp.where(lower, log_h3, log_h2))


def negative_log_ei(f: Profile, surrogate: GaussianProcess, amplitude: Scalar) -> Scalar:
    """Score a candidate as the peak-limited profile the controller will apply."""
    mu, cov = surrogate.predict(limit_peak(f, amplitude))
    return -log_expected_improvement(mu.squeeze(), cov.squeeze() ** 0.5, surrogate.ys.min())


# endregion
################################################################################


################################################################################
# region Optimization


def optimize_restarts(
    loss: Callable[[PyTree], Scalar],
    candidates: PyTree,
    bounds: tuple[PyTree, PyTree],
    max_restarts: int = 5,
) -> PyTree:
    """Screen the candidates, then refine the best ones with L-BFGS-B."""
    # only keep the best initial candidates
    best = jnp.argsort(jax.vmap(loss)(candidates))[:max_restarts]
    starts = jax.tree.map(lambda c: c[best], candidates)

    # optimize every start in parallel and keep the best local optimum
    results = jax.vmap(lambda x0: vlse.optim.minimise(loss, x0, bounds=bounds))(starts)
    return jax.tree.map(lambda a: a[jnp.argmin(results.f)], results.x)


# endregion
################################################################################
