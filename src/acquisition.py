"""Acquisition functions and the multi-restart strategy used to optimize them."""

from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy as sp
from jaxtyping import Array, Float, Scalar

jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))


@jax.jit
def log_expected_improvement(mu: Scalar, sigma: Scalar, y_best: Scalar) -> Scalar:
    # numerically stable version following https://arxiv.org/pdf/2310.20708:
    z = (y_best - mu) / sigma

    # use lax.cond to avoid propagating NaNs in the gradients
    branch1 = lambda: jnp.log(z * jsp.stats.norm.cdf(z) + jsp.stats.norm.pdf(z))
    branch2a = lambda: -2 * jnp.log(-z)
    branch2b = lambda: jax.nn.log1mexp(
        -jnp.log(-z) - jsp.stats.norm.logsf(-z) - z**2 / 2 - jnp.log(2 * jnp.pi) / 2.0
    )
    branch2 = lambda: (
        -(z**2) / 2
        - jnp.log(2 * jnp.pi) / 2
        + jax.lax.cond(z < -1 / jnp.sqrt(EPS), branch2a, branch2b)
    )
    ei = jnp.log(sigma) + jax.lax.cond(z > -1, branch1, branch2)
    return ei


@jax.jit
def upper_confidence_bound(mu: Scalar, sigma: Scalar, beta: Scalar) -> Scalar:
    return -mu + jnp.sqrt(beta) * sigma


def optimize_restarts(
    acquisition_loss: Callable[[Float[Array, "p"]], tuple[Scalar, Float[Array, "p"]]],
    candidates: Float[Array, "n p"],
    max_restarts: int = 5,
    options: dict = dict(maxiter=100, ftol=EPS, gtol=0.0),
    screening_loss: Callable[[Float[Array, "n p"]], Float[Array, "n"]] | None = None,
    domain: tuple[float, float] = (0.0, 1.0),
) -> Float[Array, "p"]:
    """Screen candidates by loss, then run L-BFGS-B from the best few.

    ``acquisition_loss`` returns (value, gradient) and is minimized over
    ``domain`` in every coordinate. ``screening_loss`` is an optional batched,
    value-only version used for the screening pass, which is the bulk of the
    candidate evaluations. Returns the location of the best local optimum found.
    """
    # only keep the best initial candidates
    if screening_loss is not None:
        losses = screening_loss(candidates)
    else:
        losses = [acquisition_loss(c)[0] for c in candidates]
    candidates = candidates[np.argsort(losses)[:max_restarts]]

    results = [
        sp.optimize.minimize(
            fun=acquisition_loss,
            x0=c,
            jac=True,
            method="L-BFGS-B",
            bounds=[domain] * len(c),
            options=options,
        )
        for c in candidates
    ]

    losses = jnp.array([result.fun for result in results])
    return jnp.array(results[int(jnp.argmin(losses))].x)
