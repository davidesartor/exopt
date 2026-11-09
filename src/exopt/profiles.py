from typing import NamedTuple, Self
from jaxtyping import Array, Float

import jax
import jax.numpy as jnp

HARMONICS = 64  # truncation of the Fourier series, beyond the constant term
RHO_RANGE = (0.3, 2.5)  # atom lengthscales [rad of gait phase]
MAX_WEIGHT = 0.5  # atom peak torque bound, as a fraction of MAX_TORQUE


def harmonics() -> Float[Array, "n"]:
    """Harmonics 0..HARMONICS of the gait cycle."""
    return jnp.arange(HARMONICS + 1, dtype=float)


def atom_spectrum(rho: Float[Array, "..."]) -> Float[Array, "... n"]:
    """Cosine coefficients of a periodized RBF atom, scaled to a peak of 1 at its center.

    By Poisson summation, sum_m exp(-(t + 2pi m)^2 / 2rho^2) is proportional to
    1 + 2 sum_{n>=1} exp(-(rho n)^2 / 2) cos(n t). Its peak is at its center t = 0,
    where the series is the sum of its coefficients.
    """
    n = harmonics()
    a = jnp.where(n == 0, 1.0, 2.0) * jnp.exp(-0.5 * (jnp.asarray(rho)[..., None] * n) ** 2)
    return a / a.sum(-1, keepdims=True)


def atom(rho: Float[Array, "..."], lag: Float[Array, "..."]) -> Float[Array, "..."]:
    """Periodized RBF atom of peak 1, at a lag [rad] from its center."""
    return (atom_spectrum(rho) * jnp.cos(harmonics() * jnp.asarray(lag)[..., None])).sum(-1)


class Profile(NamedTuple):
    """Torque profile: sum_k weights_k atom(phase - centers_k; rho), each atom of peak 1."""

    centers: Float[Array, "... k"]
    weights: Float[Array, "... k"]
    rho: Float[Array, "..."]

    @property
    def atoms(self) -> int:
        return self.centers.shape[-1]

    def series(self) -> tuple[Float[Array, "... n"], Float[Array, "... n"]]:
        """Fourier coefficients: f(phase) = sum_n cos_n cos(n phase) + sin_n sin(n phase)."""
        a = atom_spectrum(self.rho)[..., None, :]
        angle = harmonics() * self.centers[..., :, None]
        w = self.weights[..., :, None]
        return (w * a * jnp.cos(angle)).sum(-2), (w * a * jnp.sin(angle)).sum(-2)

    @jax.jit
    def __call__(self, phase: Float[Array, "..."]) -> Float[Array, "..."]:
        cos, sin = self.series()
        angle = harmonics() * jnp.asarray(phase)[..., None]
        return (cos * jnp.cos(angle) + sin * jnp.sin(angle)).sum(-1)

    def pad_to(self, atoms: int) -> Self:
        """Zero-weight atoms change nothing, so padding is exact."""
        pad = lambda c: jnp.pad(c, [(0, 0)] * (c.ndim - 1) + [(0, atoms - c.shape[-1])])
        return self._replace(centers=pad(self.centers), weights=pad(self.weights))

    def to_dict(self) -> dict:
        floats = lambda a: [float(c) for c in a]
        return dict(centers=floats(self.centers), weights=floats(self.weights), rho=float(self.rho))

    @classmethod
    def from_dict(cls, payload: dict) -> Self:
        return cls(*(jnp.asarray(payload[k], dtype=float) for k in cls._fields))

    @classmethod
    def stack(cls, profiles: list[Self]) -> Self:
        """One batched Profile, padded to the largest atom count."""
        k = max(p.atoms for p in profiles)
        return jax.tree.map(lambda *leaves: jnp.stack(leaves), *(p.pad_to(k) for p in profiles))


def features(f: Profile) -> Float[Array, "... 2n+1"]:
    """Fourier coefficients scaled so their euclidean distance is the L2 distance over the cycle."""
    # cos(n.) and sin(n.) have squared norm pi over the cycle, the constant 2pi
    cos, sin = f.series()
    norm = jnp.sqrt(jnp.pi * jnp.where(harmonics() == 0, 2.0, 1.0))
    return jnp.concatenate([norm * cos, (norm * sin)[..., 1:]], -1)


def coefficients(f: Profile) -> Float[Array, "... 3"]:
    """A single atom's center, weight and rho, each mapped from its search range to the unit interval."""
    assert f.atoms == 1, "coefficient embedding needs single-atom profiles"
    unit_center = f.centers[..., 0] / (2 * jnp.pi)
    unit_weight = (f.weights[..., 0] + MAX_WEIGHT) / (2 * MAX_WEIGHT)
    unit_rho = (f.rho - RHO_RANGE[0]) / (RHO_RANGE[1] - RHO_RANGE[0])
    return jnp.stack([unit_center, unit_weight, unit_rho], -1)


# GP inputs: L2 geometry of the functions, or a vanilla box of atom parameters
EMBEDDINGS = dict(functional=features, coefficients=coefficients)


