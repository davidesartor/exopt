from typing import NamedTuple, Self
from jaxtyping import Array, Float, Scalar
import warnings

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
import numpy as np
import scipy as sp

from . import kernels

jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))


def auto_bounds(
    xs: Float[Array, "n d"],
    min_cor: float = 0.01,
    max_cor: float = 0.5,
    p: float = 0.05,
) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
    xs = np.asarray(xs)
    lo, hi = xs.min(axis=0), xs.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    scaled = (xs - lo) / span

    # squared euclidean distances between distinct design points
    unique = np.unique(scaled, axis=0)
    diff = unique[:, None, :] - unique[None, :, :]
    d2_full = (diff**2).sum(axis=-1)
    d2 = d2_full[np.triu_indices(len(unique), k=1)]
    if d2.size == 0:  # <2 distinct points: nothing to learn, use a broad range
        return jnp.full(xs.shape[1], EPS), jnp.asarray(np.maximum(span, 1.0))

    repr_low_dist = np.quantile(d2, p)
    repr_lar_dist = np.quantile(d2, 1 - p)
    theta_min = -repr_low_dist / np.log(min_cor)
    theta_max = -repr_lar_dist / np.log(max_cor)

    lower = np.maximum(np.sqrt(theta_min / 2) * span, EPS)
    upper = np.maximum(np.sqrt(theta_max / 2) * span, lower + EPS)
    return jnp.asarray(lower), jnp.asarray(upper)


class Module(eqx.Module):
    def _replace(self, **kwargs) -> Self:
        where = lambda m: tuple(getattr(m, k) for k in kwargs.keys())
        return eqx.tree_at(where, self, kwargs.values(), is_leaf=lambda x: x is None)


class Gaussian(NamedTuple):
    mean: Float[Array, "n"]
    cov: Float[Array, "n n"]


@jax.jit
def gp_posterior(
    Kxx: Float[Array, "m m"],
    Kox: Float[Array, "n m"],
    Koo: Float[Array, "n n"],
    observed_ys: Float[Array, "n"],
    b: Scalar,
) -> Gaussian:
    # posterior mean and covariance
    gain = jnp.linalg.solve(Koo, Kox).T
    mean = b + gain @ (observed_ys - b)
    cov = Kxx - gain @ Kox

    # Add correction based on the trend estimation correlation
    Kbx = jnp.ones((1, len(observed_ys))) @ gain.T
    cov = cov + (1 - Kbx).T @ (1 - Kbx) / jnp.linalg.inv(Koo).sum()
    return Gaussian(mean=mean, cov=cov)


@jax.jit
def loglikelihood(
    Koo: Float[Array, "n n"],
    ys: Float[Array, "n"],
) -> tuple[Scalar, Scalar, Scalar]:
    # cholesky of K and compute logdet
    K_sqrt, is_lower = jsp.linalg.cho_factor(Koo)
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(K_sqrt)))

    # compute Ki_1=(K^-1 @ 1) and Ki_y=(K^-1 @ y)
    Ki_1, Ki_y = jsp.linalg.cho_solve(
        c_and_lower=(K_sqrt, is_lower),
        b=jnp.stack([jnp.ones_like(ys), ys], 1),
    ).T

    # compute optimal trend b and scale nu
    b = (Ki_1 * ys).sum() / Ki_1.sum()
    nu = jnp.dot((ys - b) / len(ys), (Ki_y - Ki_1 * b))

    # likelihood when marginalizing over trend and variance
    loglik = -0.5 * (len(ys) * jnp.log(nu) + logdetK)
    return (loglik, b, nu)


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


class GaussianProcess(Module):
    # kernel definition
    metric: kernels.Metric = kernels.Euclidean()
    profile: kernels.Profile = kernels.SquaredExponential()

    # model parameters
    rho: Float[Array, "d"] = eqx.field(default=None)
    g: Scalar = eqx.field(default=None)
    nu: Scalar = eqx.field(default=None)
    b: Scalar = eqx.field(default=None)

    # observed data
    observed_xs: Float[Array, "n d"] = eqx.field(default=None)
    observed_ys: Float[Array, "n"] = eqx.field(default=None)

    # cached covariance matrix of the observed ys
    Koo: Float[Array, "n n"] = eqx.field(default=None)

    @eqx.filter_jit
    def kernel(
        self,
        rho: Float[Array, "d"],
        xs1: Float[Array, "m d"],
        xs2: Float[Array, "n d"],
    ) -> Float[Array, "m n"]:
        return self.profile(self.metric(rho, xs1, xs2))

    @eqx.filter_jit
    def predict(self, xs: Float[Array, "m d"]) -> Gaussian:
        # compute covariance matrices
        Kxx = self.nu * self.kernel(self.rho, xs, xs)
        Kox = self.nu * self.kernel(self.rho, self.observed_xs, xs)
        Koo = self.nu * self.Koo
        return gp_posterior(Kxx, Kox, Koo, self.observed_ys, self.b)

    def fit(
        self,
        xs: Float[Array, "n d"],
        ys: Float[Array, "n"],
        *,
        warmstart: bool = False,
        nugget_range: tuple[float, float] = (EPS, 1e-3),
        max_iterations: int = 100,
        ftol: float = EPS,
        gtol: float = 0.0,
    ) -> Self:
        @jax.jit
        @jax.value_and_grad
        def mle_loss(params: Float[Array, "d+1"]):
            rho, g = params[:-1], params[-1]
            Koo = self.kernel(rho, xs, xs) + g * jnp.eye(len(ys))
            loglik, b, nu = loglikelihood(Koo, ys)
            return -loglik

        def verbose_loss(params: Float[Array, "d+1"]):
            val, grad = mle_loss(params)
            if jnp.isnan(val) or jnp.isnan(grad).any():
                warnings.warn(f"NaN detected in loss or gradient: {params}")
            return val, grad

        # per-dimension lengthscale bounds, data-driven (hetGP auto_bounds)
        n, d = xs.shape
        ls_lower, ls_upper = auto_bounds(xs)

        # initialization (hetGP defaults for the auto_bounds path):
        #   lengthscale at the geometric mean of its bounds, nugget at 0.1
        #   (the no-replicates case), clamped to the allowed nugget range
        nugget = min(0.1, nugget_range[1])
        lengthscale = jnp.sqrt(ls_lower * ls_upper)
        if warmstart:
            nugget = self.g if self.g is not None else nugget
            lengthscale = self.rho if self.rho is not None else lengthscale
        init_params = jnp.concatenate([jnp.broadcast_to(lengthscale, (d,)), jnp.array([nugget])])

        # run optimization
        bounds = [(float(lo), float(hi)) for lo, hi in zip(ls_lower, ls_upper)]
        result = sp.optimize.minimize(
            fun=verbose_loss,
            x0=init_params,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds + [nugget_range],
            options=dict(maxiter=max_iterations, ftol=ftol, gtol=gtol),
        )

        # extract the optimal parameters and infer the rest
        rho = jnp.array(result.x[:-1])
        g = jnp.array(result.x[-1])
        Koo = self.kernel(rho, xs, xs) + g * jnp.eye(len(ys))
        llk, b, nu = loglikelihood(Koo, ys)

        # return a new instance with the fitted parameters and observed data
        return self._replace(
            rho=rho, g=g, nu=nu, b=b, Koo=Koo, observed_xs=xs, observed_ys=ys
        )
