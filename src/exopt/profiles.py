"""Torque profiles as Fourier coefficients and their geometry."""

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

    def to_dict(self) -> dict:
        """JSON-safe coefficients of a single profile."""
        return dict(sin=[float(c) for c in self.sin], cos=[float(c) for c in self.cos])

    @classmethod
    def from_dict(cls, payload: dict) -> Self:
        return cls(jnp.asarray(payload["sin"]), jnp.asarray(payload["cos"]))

    @classmethod
    def stack(cls, profiles: list[Self]) -> Self:
        """One batched Profile, zero-padded to the largest harmonic count."""
        H = max(p.harmonics for p in profiles)
        padded = [p.pad_to(H) for p in profiles]
        return cls(jnp.stack([p.sin for p in padded]), jnp.stack([p.cos for p in padded]))


def spectrum(harmonics: int, sobolev_order: float = 1.0) -> Float[Array, "H"]:
    """Sobolev kernel eigenvalues m^(-2s) on harmonics 1..H."""
    m = jnp.arange(1, harmonics + 1)
    return m ** (-2.0 * sobolev_order)


def squared_differences(f: Profile, g: Profile) -> Float[Array, "... H"]:
    """Per-harmonic squared coefficient differences between two Profiles."""
    return (f.sin - g.sin) ** 2 + (f.cos - g.cos) ** 2
