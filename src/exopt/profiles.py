from typing import NamedTuple, Self
from jaxtyping import Array, Float

import jax
import jax.numpy as jnp

from exopt import kernels

ATOM_KERNEL = kernels.SquaredExponential()  # periodized RBF atoms
RHO_RANGE = (0.3, 2.5)  # atom lengthscales [rad of gait phase]


################################################################################
# region Profiles


class Profile(NamedTuple):
    """Zero-mean torque profile: sum_k weights_k k_per(phase - centers_k; rho)."""

    centers: Float[Array, "... k"]
    weights: Float[Array, "... k"]
    rho: Float[Array, "..."]

    @property
    def atoms(self) -> int:
        return self.centers.shape[-1]

    @jax.jit
    def __call__(self, phase: Float[Array, "..."]) -> Float[Array, "..."]:
        lags = jnp.asarray(phase)[..., None] - self.centers
        k = kernels.periodic(ATOM_KERNEL, jnp.asarray(self.rho)[..., None], lags)
        return (self.weights * k).sum(-1)

    def pad_to(self, atoms: int) -> Self:
        """Zero-weight atoms change nothing, so padding is exact."""
        pad = lambda c: jnp.pad(c, [(0, 0)] * (c.ndim - 1) + [(0, atoms - c.shape[-1])])
        return self._replace(centers=pad(self.centers), weights=pad(self.weights))

    def scale(self, factor: Float[Array, "..."]) -> Self:
        return self._replace(weights=factor * self.weights)

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


# endregion
################################################################################


################################################################################
# region L2 Geometry


def features(f: Profile) -> Float[Array, "... 2n"]:
    """Fourier coefficients scaled so their euclidean distance is the L2 distance."""
    # an atom has coefficients mu_n (cos nc, sin nc), and cos(n.), sin(n.) have norm^2 pi
    n = kernels.SPECTRAL_HARMONICS
    gain = jnp.sqrt(jnp.pi) * jnp.exp(kernels.log_eigenvalues(ATOM_KERNEL, f.rho, n))
    angle = kernels.harmonics(n) * f.centers[..., :, None]
    sin = gain * (f.weights[..., :, None] * jnp.sin(angle)).sum(-2)
    cos = gain * (f.weights[..., :, None] * jnp.cos(angle)).sum(-2)
    return jnp.concatenate([sin, cos], -1)


def squared_distances(fs: Profile, gs: Profile) -> Float[Array, "n m"]:
    zf, zg = features(fs), features(gs)
    d2 = (zf**2).sum(-1)[:, None] + (zg**2).sum(-1)[None, :] - 2 * zf @ zg.T
    return jnp.maximum(d2, 0.0)


# endregion
################################################################################
