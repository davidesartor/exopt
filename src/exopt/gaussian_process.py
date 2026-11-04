from typing import NamedTuple
from jaxtyping import Array, Float, Scalar

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import vlse

from exopt.profiles import Profile, squared_distances

NUGGET_RANGE = (1e-2, 1e2)
LENGTHSCALE_RANGE = (1e-1, 1e1)  # relative to the median distance between observations


@jax.jit
def loglikelihood(Koo: Float[Array, "n n"], ys: Float[Array, "n"]) -> tuple[Scalar, Scalar, Scalar]:
    # cholesky of K and compute logdet
    K_sqrt, is_lower = jsp.linalg.cho_factor(Koo)
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(K_sqrt)))

    # compute Ki_1=(K^-1 @ 1) and Ki_y=(K^-1 @ y)
    Ki_1, Ki_y = jsp.linalg.cho_solve((K_sqrt, is_lower), jnp.stack([jnp.ones_like(ys), ys], 1)).T

    # compute optimal trend b and scale nu
    b = (Ki_1 * ys).sum() / Ki_1.sum()
    nu = jnp.dot((ys - b) / len(ys), Ki_y - Ki_1 * b)

    # likelihood when marginalizing over trend and variance
    loglik = -0.5 * (len(ys) * jnp.log(nu) + logdetK)
    return loglik, b, nu


def correlation(lengthscale: Scalar, fs: Profile, gs: Profile) -> Float[Array, "n m"]:
    """Squared exponential in the L2 distance between profiles."""
    return jnp.exp(-0.5 * squared_distances(fs, gs) / lengthscale**2)


def mle_loss(log_params: Float[Array, "2"], xs: Profile, ys: Float[Array, "n"]) -> Scalar:
    lengthscale, nugget = jnp.exp(log_params)
    Koo = correlation(lengthscale, xs, xs) + nugget * jnp.eye(len(ys))
    return -loglikelihood(Koo, ys)[0]


class GaussianProcess(NamedTuple):
    # model parameters
    lengthscale: Scalar
    nugget: Scalar
    b: Scalar
    nu: Scalar

    # observed data and cached factorizations
    xs: Profile
    ys: Float[Array, "n"]
    Koo_sqrt: Float[Array, "n n"]
    Koo_inv_sum: Scalar

    @staticmethod
    @jax.jit
    def fit(xs: Profile, ys: Float[Array, "n"]) -> "GaussianProcess":
        # lengthscale range around the median distance between observations
        d2 = squared_distances(xs, xs)
        median = jnp.sqrt(jnp.nanmedian(jnp.where(jnp.eye(len(ys), dtype=bool), jnp.nan, d2)))
        median = jnp.where(jnp.isfinite(median) & (median > 0), median, 1.0)

        # maximize the likelihood over log(lengthscale, nugget)
        bounds = (
            jnp.log(jnp.array([median * LENGTHSCALE_RANGE[0], NUGGET_RANGE[0]])),
            jnp.log(jnp.array([median * LENGTHSCALE_RANGE[1], NUGGET_RANGE[1]])),
        )
        x0 = jnp.log(jnp.array([median, 0.1 * NUGGET_RANGE[1]]))
        result = vlse.optim.minimise(mle_loss, x0, bounds=bounds, args=(xs, ys))

        # cache what predict needs
        lengthscale, nugget = jnp.exp(result.x)
        Koo = correlation(lengthscale, xs, xs) + nugget * jnp.eye(len(ys))
        _, b, nu = loglikelihood(Koo, ys)
        Koo_sqrt = jsp.linalg.cho_factor(Koo, lower=True)[0]
        Koo_inv_sum = jsp.linalg.cho_solve((Koo_sqrt, True), jnp.ones_like(ys)).sum()
        return GaussianProcess(lengthscale, nugget, b, nu, xs, ys, Koo_sqrt, Koo_inv_sum)

    @jax.jit
    def predict(self, xs: Profile) -> tuple[Float[Array, "... m"], Float[Array, "... m m"]]:
        """Joint posterior over the trailing stack axis; extra leading axes batch."""
        if xs.centers.ndim == 1:
            xs = jax.tree.map(lambda a: a[None], xs)
        if xs.centers.ndim > 2:
            return jax.vmap(self.predict)(xs)

        # posterior mean and covariance
        Kxx = correlation(self.lengthscale, xs, xs)
        Kox = correlation(self.lengthscale, self.xs, xs)
        gain = jsp.linalg.cho_solve((self.Koo_sqrt, True), Kox).T
        mean = self.b + gain @ (self.ys - self.b)
        cov = self.nu * (Kxx - gain @ Kox)

        # add correction based on the trend estimation correlation
        Kbx = jnp.ones((1, len(self.ys))) @ gain.T
        cov = cov + self.nu * (1 - Kbx).T @ (1 - Kbx) / self.Koo_inv_sum
        return mean, cov

    def leave_one_out(self) -> tuple[Float[Array, "n"], Float[Array, "n"]]:
        """Prediction of each observation from all the others (fixed hyperparameters)."""
        Koo_inv = jsp.linalg.cho_solve((self.Koo_sqrt, True), jnp.eye(len(self.ys)))
        d = jnp.diag(Koo_inv)
        mean = self.ys - Koo_inv @ (self.ys - self.b) / d
        return mean, jnp.sqrt(self.nu / d)
