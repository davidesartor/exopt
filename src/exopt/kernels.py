from typing import Protocol
from jaxtyping import Array, Float

import jax.numpy as jnp
import jax.scipy as jsp

SPECTRAL_HARMONICS = 64  # truncation of every periodic kernel series


################################################################################
# region Spectral Densities


class SpectralDensity(Protocol):
    """Log spectral density log S(omega) of a unit-variance stationary kernel."""

    def __call__(
        self, omega: Float[Array, "..."], rho: Float[Array, "..."]
    ) -> Float[Array, "..."]: ...


class SquaredExponential(SpectralDensity):
    def __call__(self, omega, rho):
        return jnp.log(rho) + 0.5 * jnp.log(2 * jnp.pi) - 0.5 * (rho * omega) ** 2


class Matern(SpectralDensity):
    def __init__(self, nu: float):
        self.nu = nu

    def __call__(self, omega, rho):
        lam = 2 * self.nu / rho**2
        return (
            0.5 * jnp.log(4 * jnp.pi)
            + jsp.special.gammaln(self.nu + 0.5)
            - jsp.special.gammaln(self.nu)
            + self.nu * jnp.log(lam)
            - (self.nu + 0.5) * jnp.log(lam + omega**2)
        )


# endregion
################################################################################


################################################################################
# region Periodization


def harmonics(n: int = SPECTRAL_HARMONICS) -> Float[Array, "n"]:
    return jnp.arange(1, n + 1, dtype=float)


def log_eigenvalues(
    density: SpectralDensity, rho: Float[Array, "..."], n: int = SPECTRAL_HARMONICS
) -> Float[Array, "... n"]:
    """Poisson summation: sum_m k(t + 2pi m) has eigenvalue S(n)/pi on cos(n.), sin(n.)."""
    return density(harmonics(n), jnp.asarray(rho)[..., None]) - jnp.log(jnp.pi)


def periodic(
    density: SpectralDensity, rho: Float[Array, "..."], lag: Float[Array, "..."]
) -> Float[Array, "..."]:
    """Periodized kernel at lag [rad] by its Fourier series, constant term dropped."""
    lam = jnp.exp(log_eigenvalues(density, rho))
    return (lam * jnp.cos(harmonics() * jnp.asarray(lag)[..., None])).sum(-1)


# endregion
################################################################################
