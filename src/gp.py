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


CHOLESKY_LOWER = True


@jax.jit
def gp_posterior(
    Kxx: Float[Array, "m m"],
    Kox: Float[Array, "n m"],
    Koo_sqrt: Float[Array, "n n"],  # cholesky factor of the *unscaled* Koo
    Koo_inv_sum: Scalar,  # sum of the entries of the unscaled Koo inverse
    observed_ys: Float[Array, "n"],
    b: Scalar,
    nu: Scalar,
) -> Gaussian:
    """Posterior from a prefactored Koo: nu cancels in the gain and factors out of cov."""
    # posterior mean and covariance
    gain = jsp.linalg.cho_solve((Koo_sqrt, CHOLESKY_LOWER), Kox).T
    mean = b + gain @ (observed_ys - b)
    cov = nu * (Kxx - gain @ Kox)

    # Add correction based on the trend estimation correlation
    Kbx = jnp.ones((1, len(observed_ys))) @ gain.T
    cov = cov + nu * (1 - Kbx).T @ (1 - Kbx) / Koo_inv_sum
    return Gaussian(mean=mean, cov=cov)


@jax.jit
def prefactor(Koo: Float[Array, "n n"]) -> tuple[Float[Array, "n n"], Scalar]:
    """Factor Koo once per fit, so the posterior costs a triangular solve instead of O(n^3)."""
    Koo_sqrt = jsp.linalg.cho_factor(Koo, lower=CHOLESKY_LOWER)[0]
    ones = jnp.ones_like(Koo[0])
    return Koo_sqrt, jsp.linalg.cho_solve((Koo_sqrt, CHOLESKY_LOWER), ones).sum()


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

    # cached covariance matrix of the observed ys, factored once at fit
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
        # compute covariance matrices
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
        Koo_sqrt, Koo_inv_sum = prefactor(Koo)

        # return a new instance with the fitted parameters and observed data
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
    """Free function, not a closure: a closure recompiles on every refit."""
    rho, g = params[0], params[-1]
    Koo = profile(dists / jnp.sqrt(rho)) + g * jnp.eye(len(ys))
    loglik, b, nu = loglikelihood(Koo, ys)
    return -loglik


class FunctionalGaussianProcess(Module):
    """GP indexed by RKHS functions instead of vectors.

    Candidates may each carry their own lengthscale, so they are not elements
    of one common space. ``ambient`` is the space they are all compared in; its
    lengthscale must sit at or below the candidate range for the inner product
    to be finite. Everything downstream is the scalar GP with the RKHS distance
    swapped in for the distance between parameter vectors -- which is also why
    the basis size may vary between observations.
    """

    # kernel definition
    profile: kernels.Profile = kernels.SquaredExponential()

    # reference space the input functions are compared in
    ambient: rkhs.RKHS = eqx.field(default=None)

    # model parameters
    rho: Scalar = eqx.field(default=None)
    g: Scalar = eqx.field(default=None)
    nu: Scalar = eqx.field(default=None)
    b: Scalar = eqx.field(default=None)

    # observed data (padded to a common basis size when stacked)
    observed_fs: list[rkhs.Function] = eqx.field(default=None)
    observed_ys: Float[Array, "n"] = eqx.field(default=None)

    # cached covariance matrix of the observed ys, factored once at fit
    Koo: Float[Array, "n n"] = eqx.field(default=None)
    Koo_sqrt: Float[Array, "n n"] = eqx.field(default=None)
    Koo_inv_sum: Scalar = eqx.field(default=None)

    # rho-independent, so the next fit can extend it instead of rebuilding it
    dists: Float[Array, "n n"] = eqx.field(default=None)

    @eqx.filter_jit
    def metric(self, f1: rkhs.Function, f2: rkhs.Function) -> Scalar:
        return self.ambient.distance(f1, f2)

    def pairwise_distances(
        self,
        fs1: list[rkhs.Function],
        fs2: list[rkhs.Function],
    ) -> Float[Array, "m n"]:
        """Every function is padded to the same basis size, so this vmaps in one dispatch."""
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
        # compute covariance matrices
        Kxx = self.kernel(self.rho, fs, fs)
        Kox = self.kernel(self.rho, self.observed_fs, fs)
        return gp_posterior(
            Kxx, Kox, self.Koo_sqrt, self.Koo_inv_sum, self.observed_ys, self.b, self.nu
        )

    @eqx.filter_jit
    def predict_marginals(self, fs: rkhs.Function) -> Gaussian:
        """Independent scalar posteriors for a stacked batch of candidates, one dispatch.

        Screening only needs each candidate's own marginal, so this never forms the
        m x m block that predict would build across candidates.
        """
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
        """Grow a cached distance block to cover fs, computing only the new rows.

        Assumes fs[:m] are the functions the cache was built from, in order.
        Distances do not depend on rho or g, so a cache stays valid across refits.
        """
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
        # precalc the metric to speedup mle calls
        if cached_dists is None:
            dists = self.pairwise_distances(fs, fs)
        else:
            dists = self.extend_distances(fs, cached_dists)

        def verbose_loss(params: Float[Array, "2"]):
            val, grad = functional_mle_loss(params, dists, ys, self.profile)
            if jnp.isnan(val) or jnp.isnan(grad).any():
                warnings.warn(f"NaN detected in loss or gradient: {params}")
            return val, grad

        # initialization; auto_bounds does not apply here (a single, scalar
        # lengthscale on a distance that is not per-coordinate)
        nugget = min(0.1, nugget_range[1])
        lengthscale = 0.9 * lengthscale_range[1] + 0.1 * lengthscale_range[0]
        if warmstart:
            nugget = self.g if self.g is not None else nugget
            lengthscale = self.rho if self.rho is not None else lengthscale
        init_params = jnp.array([lengthscale, nugget])

        # run optimization
        result = sp.optimize.minimize(
            fun=verbose_loss,
            x0=init_params,
            jac=True,
            method="L-BFGS-B",
            bounds=[lengthscale_range, nugget_range],
            options=dict(maxiter=max_iterations, ftol=ftol, gtol=gtol),
        )

        # extract the optimal parameters and infer the rest
        rho = jnp.array(result.x[0])
        g = jnp.array(result.x[-1])
        Koo = self.profile(dists / jnp.sqrt(rho)) + g * jnp.eye(len(ys))
        llk, b, nu = loglikelihood(Koo, ys)
        Koo_sqrt, Koo_inv_sum = prefactor(Koo)

        # return a new instance with the fitted parameters and observed data
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
