from typing import NamedTuple
from jaxtyping import Array, Float, Scalar

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import vlse


LENGTHSCALE_RANGE = (1e-1, 1e1)  # relative to the median distance between observations
SIGNAL_RANGE = (1e-2, 1e2)  # signal variance, relative to the variance of the observations
NUGGET_RANGE = (1e-4, 1e1)  # variability beyond the measurement SE, relative to the same


def squared_distances(xs: Float[Array, "n d"], zs: Float[Array, "m d"]) -> Float[Array, "n m"]:
    d2 = (xs**2).sum(-1)[:, None] + (zs**2).sum(-1)[None, :] - 2 * xs @ zs.T
    return jnp.maximum(d2, 0.0)


def correlation(lengthscale: Scalar, xs: Float[Array, "n d"], zs: Float[Array, "m d"]) -> Float[Array, "n m"]:
    """Squared exponential in the euclidean distance between embedded profiles."""
    return jnp.exp(-0.5 * squared_distances(xs, zs) / lengthscale**2)


def covariance(params: Float[Array, "3"], xs: Float[Array, "n d"], noise: Float[Array, "n"]) -> Float[Array, "n n"]:
    """Covariance of the observations: signal, plus a nugget, plus each measurement's own variance."""
    lengthscale, signal, nugget = params
    return signal * correlation(lengthscale, xs, xs) + jnp.diag(nugget + noise)


@jax.jit
def restricted_loglikelihood(C: Float[Array, "n n"], ys: Float[Array, "n"]) -> tuple[Scalar, Scalar]:
    """Log-likelihood with the constant trend integrated out (REML), and the trend's GLS estimate."""
    C_sqrt = jnp.linalg.cholesky(C)
    Ci_1, Ci_y = jsp.linalg.cho_solve((C_sqrt, True), jnp.stack([jnp.ones_like(ys), ys], 1)).T
    sum_1 = Ci_1.sum()
    b = Ci_y.sum() / sum_1
    quadratic = ys @ Ci_y - b**2 * sum_1  # (y - b)' C^-1 (y - b)
    logdet = 2.0 * jnp.log(jnp.diag(C_sqrt)).sum()
    return -0.5 * (logdet + quadratic + jnp.log(sum_1)), b


def mle_loss(log_params: Float[Array, "3"], xs: Float[Array, "n d"], ys: Float[Array, "n"], noise) -> Scalar:
    return -restricted_loglikelihood(covariance(jnp.exp(log_params), xs, noise), ys)[0]


class GaussianProcess(NamedTuple):
    """GP on embedded profiles, observed as measurement means with known variances `noise`."""

    # model parameters
    lengthscale: Scalar
    signal: Scalar  # variance of the loss across profiles
    nugget: Scalar  # variance of a measurement beyond its SE (drift between segments)
    b: Scalar

    # observed data and cached factorizations
    xs: Float[Array, "n d"]
    ys: Float[Array, "n"]
    noise: Float[Array, "n"]
    C_sqrt: Float[Array, "n n"]
    C_inv_sum: Scalar

    @staticmethod
    @jax.jit
    def fit(xs: Float[Array, "n d"], ys: Float[Array, "n"], noise: Float[Array, "n"]) -> "GaussianProcess":
        """Fit lengthscale, signal variance and nugget by restricted maximum likelihood."""
        # lengthscale range around the median distance between observations
        d2 = squared_distances(xs, xs)
        median = jnp.sqrt(jnp.nanmedian(jnp.where(jnp.eye(len(ys), dtype=bool), jnp.nan, d2)))
        median = jnp.where(jnp.isfinite(median) & (median > 0), median, 1.0)

        # variance ranges around the spread of the observations
        scale = jnp.var(ys) + noise.mean()
        scale = jnp.where(scale > 0, scale, 1.0)

        lower = jnp.array([median * LENGTHSCALE_RANGE[0], scale * SIGNAL_RANGE[0], scale * NUGGET_RANGE[0]])
        upper = jnp.array([median * LENGTHSCALE_RANGE[1], scale * SIGNAL_RANGE[1], scale * NUGGET_RANGE[1]])
        x0 = jnp.log(jnp.array([median, scale, 0.1 * scale]))
        result = vlse.optim.minimise(mle_loss, x0, bounds=(jnp.log(lower), jnp.log(upper)), args=(xs, ys, noise))

        # cache what predict needs
        params = jnp.exp(result.x)
        C = covariance(params, xs, noise)
        _, b = restricted_loglikelihood(C, ys)
        C_sqrt = jnp.linalg.cholesky(C)
        C_inv_sum = jsp.linalg.cho_solve((C_sqrt, True), jnp.ones_like(ys)).sum()
        return GaussianProcess(*params, b, xs, ys, noise, C_sqrt, C_inv_sum)

    @jax.jit
    def predict(self, xs: Float[Array, "... m d"]) -> tuple[Float[Array, "... m"], Float[Array, "... m m"]]:
        """Joint posterior of the loss (without measurement noise) over the trailing stack axis."""
        if xs.ndim == 1:
            xs = xs[None]
        if xs.ndim > 2:
            return jax.vmap(self.predict)(xs)

        # posterior mean and covariance
        Kxx = self.signal * correlation(self.lengthscale, xs, xs)
        Kox = self.signal * correlation(self.lengthscale, self.xs, xs)
        gain = jsp.linalg.cho_solve((self.C_sqrt, True), Kox).T
        mean = self.b + gain @ (self.ys - self.b)
        cov = Kxx - gain @ Kox

        # add the uncertainty of the trend estimate
        residual = 1 - gain.sum(-1, keepdims=True)
        return mean, cov + residual @ residual.T / self.C_inv_sum

    def leave_one_out(self) -> tuple[Float[Array, "n"], Float[Array, "n"]]:
        """Prediction of each measurement from all the others (fixed hyperparameters)."""
        C_inv = jsp.linalg.cho_solve((self.C_sqrt, True), jnp.eye(len(self.ys)))
        d = jnp.diag(C_inv)
        return self.ys - C_inv @ (self.ys - self.b) / d, jnp.sqrt(1 / d)
