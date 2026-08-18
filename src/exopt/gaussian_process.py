"""Distance-based GP surrogate with estimated constant mean."""

from typing import NamedTuple
from jaxtyping import Array, Float, Scalar

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import vlse


from exopt import rkhs_functions
from exopt.rkhs_functions import Profile


@jax.jit
def loglikelihood(
    Koo: Float[Array, "n n"], y: Float[Array, "n"]
) -> tuple[Scalar, Scalar, Scalar]:
    """Profile log-likelihood with mean b and scale nu solved in closed form."""
    K_sqrt, is_lower = jsp.linalg.cho_factor(Koo)
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(K_sqrt)))

    # solve K^-1 @ [1, y] in one factorized triangular solve
    Ki_1, Ki_y = jsp.linalg.cho_solve(
        c_and_lower=(K_sqrt, is_lower),
        b=jnp.stack([jnp.ones_like(y), y], 1),
    ).T

    # closed-form estimates of the constant mean and the signal scale
    b = (Ki_1 * y).sum() / Ki_1.sum()
    nu = jnp.dot((y - b) / len(y), (Ki_y - Ki_1 * b))

    loglik = -0.5 * (len(y) * jnp.log(nu) + logdetK)
    return loglik, b, nu


def correlation_matrix(
    weights: Float[Array, "H"], fs: Profile, gs: Profile
) -> Float[Array, "n m"]:
    """Squared-exponential correlations under the per-harmonic metric."""
    sqd = rkhs_functions.squared_differences(
        Profile(fs.sin[:, None], fs.cos[:, None]), gs
    )
    return jnp.exp(-0.5 * rkhs_functions.weighted_sq_distances(sqd, weights))


class GaussianProcess(NamedTuple):
    """Fitted surrogate: MLE hyperparameters plus cached training factorizations."""

    weights: Float[Array, "H"]
    nugget: Scalar
    b: Scalar
    nu: Scalar
    x: Profile
    y: Float[Array, "n"]
    Koo_sqrt: Float[Array, "n n"]
    Koo_inv_sum: Scalar

    @staticmethod
    @jax.jit
    def fit(
        x: Profile,
        y: Float[Array, "n"],
        *,
        nugget_range: tuple[float, float] = (1e-2, 1e2),
        weight_range: tuple[float, float] = (1e-2, 1e2),
    ):
        """MLE of per-harmonic weights and nugget, then cache the factorizations."""

        def loss(log_params: Float[Array, "H+1"]) -> Scalar:
            params = jnp.exp(log_params)
            Koo = correlation_matrix(params[:-1], x, x)
            return -loglikelihood(Koo + params[-1] * jnp.eye(len(y)), y)[0]

        # init weights from the Sobolev decay
        w0 = 1.0 / rkhs_functions.spectrum(x.harmonics)
        x0 = jnp.log(jnp.append(w0, 0.1 * nugget_range[1]))

        # maximize the likelihood over log(weights, nugget) in the box
        bounds = (
            jnp.log(jnp.append(w0 * weight_range[0], nugget_range[0])),
            jnp.log(jnp.append(w0 * weight_range[1], nugget_range[1])),
        )
        result = vlse.optim.minimise(loss, x0, bounds=bounds)

        # refit at the optimum and cache what predict() needs
        weights, nugget = jnp.exp(result.x[:-1]), jnp.exp(result.x[-1])
        Koo = correlation_matrix(weights, x, x) + nugget * jnp.eye(len(y))
        _, b, nu = loglikelihood(Koo, y)
        Koo_sqrt = jsp.linalg.cho_factor(Koo, lower=True)[0]
        Koo_inv_sum = jsp.linalg.cho_solve((Koo_sqrt, True), jnp.ones_like(y)).sum()
        return GaussianProcess(weights, nugget, b, nu, x, y, Koo_sqrt, Koo_inv_sum)

    @jax.jit
    def predict(
        self, x: Profile
    ) -> tuple[Float[Array, "... m"], Float[Array, "... m m"]]:
        """Joint posterior over the trailing stack axis; extra leading axes batch."""
        x = Profile(jnp.atleast_2d(x.sin), jnp.atleast_2d(x.cos))
        if x.sin.ndim > 2:
            return jax.vmap(self.predict)(x)
        
        Kxx = correlation_matrix(self.weights, x, x)
        Kox = correlation_matrix(self.weights, self.x, x)
        gain = jsp.linalg.cho_solve((self.Koo_sqrt, True), Kox).T
        mean = self.b + gain @ (self.y - self.b)
        cov = self.nu * (Kxx - gain @ Kox)

        # uncertainty of the estimated constant mean
        Kbx = jnp.ones((1, len(self.y))) @ gain.T
        cov = cov + self.nu * (1 - Kbx).T @ (1 - Kbx) / self.Koo_inv_sum
        return mean, cov
