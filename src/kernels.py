from typing import Literal, Protocol
from jaxtyping import Array, Float, Scalar

import jax
import jax.numpy as jnp
import jax.scipy as jsp


################################################################################
# region Kernel Profiles


class Profile(Protocol):
    def __call__(self, d: Float[Array, "..."]) -> Float[Array, "..."]: ...


class SquaredExponential(Profile):
    def __call__(self, d: Float[Array, "..."]) -> Float[Array, "..."]:
        k = jnp.exp(-0.5 * d**2)
        return k


class Matern(Profile):
    def __init__(self, nu: float):
        self.nu = nu

    def __call__(self, d: Float[Array, "..."]) -> Float[Array, "..."]:
        # TODO: add support for general nu
        if self.nu == 1 / 2:
            k = jnp.exp(-d)
        elif self.nu == 3 / 2:
            k = (1 + jnp.sqrt(3) * d) * jnp.exp(-jnp.sqrt(3) * d)
        elif self.nu == 5 / 2:
            k = (1 + jnp.sqrt(5) * d + 5 / 3 * d**2) * jnp.exp(-jnp.sqrt(5) * d)
        else:
            raise ValueError(f"Unsupported nu={self.nu}")
        return k


# endregion
################################################################################


################################################################################
# region Metrics


class Metric(Protocol):
    def __call__(
        self,
        rho: Float[Array, "..."],
        x1: Float[Array, "n d"],
        x2: Float[Array, "m d"],
    ) -> Float[Array, "n m"]: ...


class Minkowski(Metric):
    def __init__(self, p: int | Literal["inf", "-inf"]):
        self.p = p

    def __call__(
        self,
        rho: Float[Array, "#d"],
        x1: Float[Array, "n d"],
        x2: Float[Array, "m d"],
    ) -> Float[Array, "n m"]:
        # define the distance function for a single pair of points
        def dist(a: Float[Array, "d"], b: Float[Array, "d"]) -> Scalar:
            v = (a - b) / rho
            # use lax.cond to avoid propagating NaNs in the gradients for v=0.0
            return jax.lax.cond(
                jnp.allclose(v, 0.0),
                lambda: 0.0,
                lambda: jax.numpy.linalg.norm(v, ord=self.p),
            )

        # vectorize the distance function over pairs
        dist = jax.vmap(dist, in_axes=(None, 0))  # vectorize over x2
        dist = jax.vmap(dist, in_axes=(0, None))  # vectorize over x1
        return dist(x1, x2)


class Euclidean(Minkowski):
    def __init__(self):
        super().__init__(p=2)


class Manhattan(Minkowski):
    def __init__(self):
        super().__init__(p=1)


class Chebyshev(Minkowski):
    def __init__(self):
        super().__init__(p="inf")


class Mahalanobis(Metric):
    def __init__(self, p: int | Literal["inf", "-inf"] = 2):
        self.p = p

    def __call__(
        self,
        rho: Float[Array, "d d"],
        x1: Float[Array, "n d"],
        x2: Float[Array, "m d"],
    ) -> Float[Array, "n m"]:
        # define the distance function for a single pair of points
        def dist(a: Float[Array, "d"], b: Float[Array, "d"]) -> Scalar:
            cov_sqrt, is_lower = jsp.linalg.cho_factor(rho)
            v = jsp.linalg.solve_triangular(cov_sqrt, a - b, lower=is_lower)
            # use lax.cond to avoid propagating NaNs in the gradients for v=0.0
            return jax.lax.cond(
                jnp.allclose(v, 0.0),
                lambda: 0.0,
                lambda: jax.numpy.linalg.norm(v, ord=self.p),
            )

        # vectorize the distance function over pairs
        dist = jax.vmap(dist, in_axes=(None, 0))  # vectorize over x2
        dist = jax.vmap(dist, in_axes=(0, None))  # vectorize over x1
        return dist(x1, x2)


# endregion
################################################################################
