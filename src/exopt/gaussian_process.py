"""Distance-based GP surrogate with estimated constant mean."""

from typing import NamedTuple, Self

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import vlse
from jaxtyping import Array, Float, Scalar

from exopt.rkhs_functions import EPS

NUGGET_RANGE = (EPS, 1e-3)


class Gaussian(NamedTuple):
    mean: Float[Array, "n"]
    cov: Float[Array, "n n"]


@jax.jit
def loglikelihood(
    Koo: Float[Array, "n n"], ys: Float[Array, "n"]
) -> tuple[Scalar, Scalar, Scalar]:
    K_sqrt, is_lower = jsp.linalg.cho_factor(Koo)
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(K_sqrt)))

    Ki_1, Ki_y = jsp.linalg.cho_solve(
        c_and_lower=(K_sqrt, is_lower),
        b=jnp.stack([jnp.ones_like(ys), ys], 1),
    ).T

    b = (Ki_1 * ys).sum() / Ki_1.sum()
    nu = jnp.dot((ys - b) / len(ys), (Ki_y - Ki_1 * b))

    loglik = -0.5 * (len(ys) * jnp.log(nu) + logdetK)
    return loglik, b, nu


@jax.jit
def gp_posterior(
    Kxx: Float[Array, "m m"],
    Kox: Float[Array, "n m"],
    Koo_sqrt: Float[Array, "n n"],
    Koo_inv_sum: Scalar,
    observed_ys: Float[Array, "n"],
    b: Scalar,
    nu: Scalar,
) -> Gaussian:
    gain = jsp.linalg.cho_solve((Koo_sqrt, True), Kox).T
    mean = b + gain @ (observed_ys - b)
    cov = nu * (Kxx - gain @ Kox)

    # uncertainty of the estimated constant mean
    Kbx = jnp.ones((1, len(observed_ys))) @ gain.T
    cov = cov + nu * (1 - Kbx).T @ (1 - Kbx) / Koo_inv_sum
    return Gaussian(mean=mean, cov=cov)


def se_kernel(dists: Float[Array, "n m"], lengthscale: Scalar) -> Float[Array, "n m"]:
    return jnp.exp(-0.5 * (dists / lengthscale) ** 2)


class GaussianProcess(NamedTuple):
    lengthscale: Scalar
    nugget: Scalar
    b: Scalar
    nu: Scalar
    observed_ys: Float[Array, "n"]
    Koo_sqrt: Float[Array, "n n"]
    Koo_inv_sum: Scalar

    @classmethod
    def fit(cls, dists: Float[Array, "n n"], ys: Float[Array, "n"]) -> Self:
        off_diag = dists[jnp.triu_indices(len(ys), k=1)]
        ls_range = (
            max(float(jnp.quantile(off_diag, 0.05)) / 3, EPS),
            max(float(jnp.quantile(off_diag, 0.95)) * 3, 10 * EPS),
        )

        def loss(params: Float[Array, "2"]) -> Scalar:
            Koo = se_kernel(dists, params[0]) + params[1] * jnp.eye(len(ys))
            return -loglikelihood(Koo, ys)[0]

        x0 = jnp.array([sum(ls_range) / 2, NUGGET_RANGE[1]])
        bounds = (
            jnp.array([ls_range[0], NUGGET_RANGE[0]]),
            jnp.array([ls_range[1], NUGGET_RANGE[1]]),
        )
        result = vlse.optim.minimise(loss, x0, bounds=bounds)

        lengthscale, nugget = result.x
        Koo = se_kernel(dists, lengthscale) + nugget * jnp.eye(len(ys))
        _, b, nu = loglikelihood(Koo, ys)
        Koo_sqrt = jsp.linalg.cho_factor(Koo, lower=True)[0]
        Koo_inv_sum = jsp.linalg.cho_solve((Koo_sqrt, True), jnp.ones_like(ys)).sum()
        return cls(lengthscale, nugget, b, nu, ys, Koo_sqrt, Koo_inv_sum)

    def predict(self, dists_ox: Float[Array, "n"]) -> Gaussian:
        """Posterior at a single point, given its distances to the observations."""
        Kxx = jnp.ones((1, 1))
        Kox = se_kernel(dists_ox[:, None], self.lengthscale)
        return gp_posterior(
            Kxx, Kox, self.Koo_sqrt, self.Koo_inv_sum, self.observed_ys, self.b, self.nu
        )
