"""Torque profiles as Fourier-RKHS atoms and their geometry."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))

SOBOLEV_ORDER = 1.0
AMPLITUDE_DOMAIN = (-1.0, 1.0)
PHASE_DOMAIN = (0.0, 1.0)


def spectrum(harmonics: int) -> Float[Array, "h"]:
    m = jnp.arange(1, harmonics + 1)
    return m ** (-2.0 * SOBOLEV_ORDER)


class Profile(NamedTuple):
    """A periodic function by its Fourier coefficients on harmonics 1..H."""

    sin: Float[Array, "... H"]
    cos: Float[Array, "... H"]

    @property
    def harmonics(self) -> int:
        return self.sin.shape[-1]

    def __call__(self, phase: Float[Array, "..."]) -> Float[Array, "..."]:
        """Evaluate at gait phase [rad]; one cycle spans 2pi."""
        angle = jnp.arange(1, self.harmonics + 1) * phase[..., None]
        return (self.sin * jnp.sin(angle) + self.cos * jnp.cos(angle)).sum(-1)

    def pad_to(self, harmonics: int) -> "Profile":
        pad = [(0, 0)] * (self.sin.ndim - 1) + [(0, harmonics - self.harmonics)]
        return Profile(jnp.pad(self.sin, pad), jnp.pad(self.cos, pad))


def from_atoms(
    amplitude: Float[Array, "... k"],
    phase: Float[Array, "... k"],
    harmonics: int,
) -> Profile:
    """Sum of kernel translates: sum_i A_i * kappa_h(t - phi_i)."""
    lam = spectrum(harmonics)
    angle = 2 * jnp.pi * jnp.arange(1, harmonics + 1) * phase[..., None]
    sin = jnp.einsum("...k,...kh->...h", amplitude, lam * jnp.sin(angle))
    cos = jnp.einsum("...k,...kh->...h", amplitude, lam * jnp.cos(angle))
    return Profile(sin, cos)


def from_vector(p: Float[Array, "2k"], harmonics: int) -> Profile:
    """Build a k-atom Profile from the flat (amplitudes, phases) optimizer vector."""
    k = len(p) // 2
    return from_atoms(p[:k], p[k:], harmonics)


def vector_bounds(k: int) -> tuple[Float[Array, "2k"], Float[Array, "2k"]]:
    """Optimizer-space box for the flat (amplitudes, phases) vector."""
    (a_lo, a_hi), (p_lo, p_hi) = AMPLITUDE_DOMAIN, PHASE_DOMAIN
    lower = jnp.concatenate([jnp.full(k, a_lo), jnp.full(k, p_lo)])
    upper = jnp.concatenate([jnp.full(k, a_hi), jnp.full(k, p_hi)])
    return lower, upper


def inner(f: Profile, g: Profile) -> Float[Array, "..."]:
    lam = spectrum(f.harmonics)
    return ((f.sin * g.sin + f.cos * g.cos) / lam).sum(-1)


def distance(f: Profile, g: Profile) -> Float[Array, "..."]:
    d = Profile(f.sin - g.sin, f.cos - g.cos)
    return jnp.sqrt(jnp.maximum(inner(d, d), 0.0))


def pairwise_distances(fs: Profile, gs: Profile) -> Float[Array, "n m"]:
    take = lambda p, i: jax.tree.map(lambda leaf: leaf[i], p)
    dist_row = jax.vmap(distance, in_axes=(None, 0))
    return jax.vmap(lambda i: dist_row(take(fs, i), gs))(jnp.arange(len(fs.sin)))
