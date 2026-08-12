from typing import NamedTuple, Self
from jaxtyping import Array, Float, Scalar
import warnings

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
import numpy as np
import scipy as sp

from . import kernels, rkhs

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

    unique = np.unique(scaled, axis=0)
    diff = unique[:, None, :] - unique[None, :, :]
    d2_full = (diff**2).sum(axis=-1)
    d2 = d2_full[np.triu_indices(len(unique), k=1)]
    if d2.size == 0:
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


CHOLESKY_LOWER = True


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
    gain = jsp.linalg.cho_solve((Koo_sqrt, CHOLESKY_LOWER), Kox).T
    mean = b + gain @ (observed_ys - b)
    cov = nu * (Kxx - gain @ Kox)

    Kbx = jnp.ones((1, len(observed_ys))) @ gain.T
    cov = cov + nu * (1 - Kbx).T @ (1 - Kbx) / Koo_inv_sum
    return Gaussian(mean=mean, cov=cov)


@jax.jit
def prefactor(Koo: Float[Array, "n n"]) -> tuple[Float[Array, "n n"], Scalar]:
    Koo_sqrt = jsp.linalg.cho_factor(Koo, lower=CHOLESKY_LOWER)[0]
    ones = jnp.ones_like(Koo[0])
    return Koo_sqrt, jsp.linalg.cho_solve((Koo_sqrt, CHOLESKY_LOWER), ones).sum()


@jax.jit
def loglikelihood(
    Koo: Float[Array, "n n"],
    ys: Float[Array, "n"],
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
    return (loglik, b, nu)


class GaussianProcess(Module):
    metric: kernels.Metric = kernels.Euclidean()
    profile: kernels.Profile = kernels.SquaredExponential()

    rho: Float[Array, "d"] = eqx.field(default=None)
    g: Scalar = eqx.field(default=None)
    nu: Scalar = eqx.field(default=None)
    b: Scalar = eqx.field(default=None)

    observed_xs: Float[Array, "n d"] = eqx.field(default=None)
    observed_ys: Float[Array, "n"] = eqx.field(default=None)

    Koo: Float[Array, "n n"] = eqx.field(default=None)
    Koo_sqrt: Float[Array, "n n"] = eqx.field(default=None)
    Koo_inv_sum: Scalar = eqx.field(default=None)

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
        Kxx = self.kernel(self.rho, xs, xs)
        Kox = self.kernel(self.rho, self.observed_xs, xs)
        return gp_posterior(
            Kxx, Kox, self.Koo_sqrt, self.Koo_inv_sum, self.observed_ys, self.b, self.nu
        )

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

        n, d = xs.shape
        ls_lower, ls_upper = auto_bounds(xs)

        nugget = min(0.1, nugget_range[1])
        lengthscale = jnp.sqrt(ls_lower * ls_upper)
        if warmstart:
            nugget = self.g if self.g is not None else nugget
            lengthscale = self.rho if self.rho is not None else lengthscale
        init_params = jnp.concatenate([jnp.broadcast_to(lengthscale, (d,)), jnp.array([nugget])])

        bounds = [(float(lo), float(hi)) for lo, hi in zip(ls_lower, ls_upper)]
        result = sp.optimize.minimize(
            fun=verbose_loss,
            x0=init_params,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds + [nugget_range],
            options=dict(maxiter=max_iterations, ftol=ftol, gtol=gtol),
        )

        rho = jnp.array(result.x[:-1])
        g = jnp.array(result.x[-1])
        Koo = self.kernel(rho, xs, xs) + g * jnp.eye(len(ys))
        llk, b, nu = loglikelihood(Koo, ys)
        Koo_sqrt, Koo_inv_sum = prefactor(Koo)

        return self._replace(
            rho=rho,
            g=g,
            nu=nu,
            b=b,
            Koo=Koo,
            Koo_sqrt=Koo_sqrt,
            Koo_inv_sum=Koo_inv_sum,
            observed_xs=xs,
            observed_ys=ys,
        )


@eqx.filter_jit
@eqx.filter_value_and_grad
def functional_mle_loss(
    params: Float[Array, "2"],
    dists: Float[Array, "n n"],
    ys: Float[Array, "n"],
    profile: kernels.Profile,
) -> Scalar:
    rho, g = params[0], params[-1]
    Koo = profile(dists / jnp.sqrt(rho)) + g * jnp.eye(len(ys))
    loglik, b, nu = loglikelihood(Koo, ys)
    return -loglik


class FunctionalGaussianProcess(Module):

    profile: kernels.Profile = kernels.SquaredExponential()

    ambient: rkhs.RKHS = eqx.field(default=None)

    rho: Scalar = eqx.field(default=None)
    g: Scalar = eqx.field(default=None)
    nu: Scalar = eqx.field(default=None)
    b: Scalar = eqx.field(default=None)

    observed_fs: list[rkhs.Function] = eqx.field(default=None)
    observed_ys: Float[Array, "n"] = eqx.field(default=None)

    Koo: Float[Array, "n n"] = eqx.field(default=None)
    Koo_sqrt: Float[Array, "n n"] = eqx.field(default=None)
    Koo_inv_sum: Scalar = eqx.field(default=None)

    dists: Float[Array, "n n"] = eqx.field(default=None)

    @eqx.filter_jit
    def metric(self, f1: rkhs.Function, f2: rkhs.Function) -> Scalar:
        return self.ambient.distance(f1, f2)

    def pairwise_distances(
        self,
        fs1: list[rkhs.Function],
        fs2: list[rkhs.Function],
    ) -> Float[Array, "m n"]:
        return self.ambient.pairwise_distances(
            rkhs.Function.stack(fs1), rkhs.Function.stack(fs2)
        )

    def kernel(
        self,
        rho: Scalar,
        fs1: list[rkhs.Function],
        fs2: list[rkhs.Function],
    ) -> Float[Array, "m n"]:
        return self.profile(self.pairwise_distances(fs1, fs2) / jnp.sqrt(rho))

    def predict(self, fs: list[rkhs.Function]) -> Gaussian:
        Kxx = self.kernel(self.rho, fs, fs)
        Kox = self.kernel(self.rho, self.observed_fs, fs)
        return gp_posterior(
            Kxx, Kox, self.Koo_sqrt, self.Koo_inv_sum, self.observed_ys, self.b, self.nu
        )

    @eqx.filter_jit
    def predict_marginals(self, fs: rkhs.Function) -> Gaussian:
        observed = rkhs.Function.stack(self.observed_fs)
        self_covariance = self.profile(jnp.zeros((1, 1)))

        def marginal(f: rkhs.Function) -> Gaussian:
            dists = jax.vmap(self.ambient.distance, (0, None))(observed, f)
            Kox = self.profile(dists / jnp.sqrt(self.rho))[:, None]
            return gp_posterior(
                self_covariance,
                Kox,
                self.Koo_sqrt,
                self.Koo_inv_sum,
                self.observed_ys,
                self.b,
                self.nu,
            )

        return jax.vmap(marginal)(fs)

    def extend_distances(
        self,
        fs: list[rkhs.Function],
        cached: Float[Array, "m m"],
    ) -> Float[Array, "n n"]:
        m = len(cached)
        old, new = fs[:m], fs[m:]
        if not new:
            return cached

        cross = self.pairwise_distances(old, new)
        return jnp.block(
            [[cached, cross], [cross.T, self.pairwise_distances(new, new)]]
        )

    def fit(
        self,
        fs: list[rkhs.Function],
        ys: Float[Array, "n"],
        *,
        cached_dists: Float[Array, "m m"] | None = None,
        warmstart: bool = False,
        lengthscale_range: tuple[float, float] = (EPS, 10.0),
        nugget_range: tuple[float, float] = (EPS, 1e-3),
        max_iterations: int = 100,
        ftol: float = EPS,
        gtol: float = 0.0,
    ) -> Self:
        if cached_dists is None:
            dists = self.pairwise_distances(fs, fs)
        else:
            dists = self.extend_distances(fs, cached_dists)

        def verbose_loss(params: Float[Array, "2"]):
            val, grad = functional_mle_loss(params, dists, ys, self.profile)
            if jnp.isnan(val) or jnp.isnan(grad).any():
                warnings.warn(f"NaN detected in loss or gradient: {params}")
            return val, grad

        nugget = min(0.1, nugget_range[1])
        lengthscale = 0.9 * lengthscale_range[1] + 0.1 * lengthscale_range[0]
        if warmstart:
            nugget = self.g if self.g is not None else nugget
            lengthscale = self.rho if self.rho is not None else lengthscale
        init_params = jnp.array([lengthscale, nugget])

        result = sp.optimize.minimize(
            fun=verbose_loss,
            x0=init_params,
            jac=True,
            method="L-BFGS-B",
            bounds=[lengthscale_range, nugget_range],
            options=dict(maxiter=max_iterations, ftol=ftol, gtol=gtol),
        )

        rho = jnp.array(result.x[0])
        g = jnp.array(result.x[-1])
        Koo = self.profile(dists / jnp.sqrt(rho)) + g * jnp.eye(len(ys))
        llk, b, nu = loglikelihood(Koo, ys)
        Koo_sqrt, Koo_inv_sum = prefactor(Koo)

        return self._replace(
            rho=rho,
            g=g,
            nu=nu,
            b=b,
            Koo=Koo,
            Koo_sqrt=Koo_sqrt,
            Koo_inv_sum=Koo_inv_sum,
            dists=dists,
            observed_fs=fs,
            observed_ys=ys,
        )
