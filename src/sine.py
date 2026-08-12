
from typing import NamedTuple, Self

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar

jax.config.update("jax_enable_x64", True)

DIM = 2
PARAM_NAMES = ("amplitude", "phase")


def curve(
    t: Float[Array, "..."], amplitude: Scalar, phase: Scalar
) -> Float[Array, "..."]:
    return amplitude * jnp.sin(2 * jnp.pi * (t - phase))


class Sine(NamedTuple):

    x: Float[Array, "2"]

    @classmethod
    def from_array(cls, x: Float[Array, "2"]) -> Self:
        return cls(jnp.asarray(x))

    @property
    def amplitude(self) -> Scalar:
        return self.x[0]

    @property
    def phase(self) -> Scalar:
        return self.x[1]

    def __call__(self, t: Float[Array, "1"]) -> Scalar:
        return curve(jnp.asarray(t).squeeze(-1), self.amplitude, self.phase)

    def sample(self, ts: Float[Array, "m 1"]) -> Float[Array, "m"]:
        return curve(jnp.asarray(ts).squeeze(-1), self.amplitude, self.phase)
