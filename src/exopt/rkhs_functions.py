"""Torque profiles as Fourier-RKHS atoms and their geometry."""

from typing import NamedTuple, Self
from jaxtyping import Array, Float

import jax
import jax.numpy as jnp



class Profile(NamedTuple):
    """A periodic function by its Fourier coefficients on harmonics 1..H."""

    sin: Float[Array, "... H"]
    cos: Float[Array, "... H"]

    @property
    def harmonics(self) -> int:
        return self.sin.shape[-1]

    @jax.jit
    def __call__(self, phase: Float[Array, "..."]) -> Float[Array, "..."]:
        """Evaluate at gait phase [rad]; one cycle spans 2pi."""
        angle = jnp.arange(1, self.harmonics + 1) * phase[..., None]
        return (self.sin * jnp.sin(angle) + self.cos * jnp.cos(angle)).sum(-1)

    def pad_to(self, harmonics: int) -> Self:
        """Zero-pad the coefficients up to a larger number of harmonics."""
        pad = [(0, 0)] * (self.sin.ndim - 1) + [(0, harmonics - self.harmonics)]
        return self._replace(sin=jnp.pad(self.sin, pad), cos=jnp.pad(self.cos, pad))


def spectrum(harmonics: int, sobolev_order: float = 1.0) -> Float[Array, "h"]:
    """Sobolev kernel eigenvalues m^(-2s) on harmonics 1..H."""
    m = jnp.arange(1, harmonics + 1)
    return m ** (-2.0 * sobolev_order)


def squared_differences(f: Profile, g: Profile) -> Float[Array, "... H"]:
    """Per-harmonic squared coefficient differences between two Profiles."""
    return (f.sin - g.sin) ** 2 + (f.cos - g.cos) ** 2


def weighted_sq_distances(
    sqd: Float[Array, "... H"], weights: Float[Array, "H"]
) -> Float[Array, "..."]:
    """Squared distance under a per-harmonic metric."""
    return jnp.maximum(sqd @ weights, 0.0)
