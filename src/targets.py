"""Functional test objectives, for exercising the loop without hardware.

``Ridge`` lifts any scalar benchmark from ``virtual_library`` to the functional
domain: the candidate profile is probed at a fixed random set of points, those
readings are mixed into d scalars, and the benchmark is evaluated on them. It
gives a functional objective with known structure and a known optimum.
"""

from typing import Callable, Protocol

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Scalar

from . import virtual_library


class FunctionalTestFunction(Protocol):
    d: int

    def __call__(self, f: Callable[[Float[Array, "d"]], Scalar]) -> Scalar: ...


class SincProjection(FunctionalTestFunction):
    """Squared distance to a sinc target, averaged over a fixed random grid.

    The functional benchmark from BOFUS: it depends on the candidate everywhere
    rather than at a few probe points, so it is the honest test of whether the
    search can shape a curve. Optimum is 0, at f == target.
    """

    def __init__(self, d: int = 1, seed: int = 0, n: int = 10000):
        self.d = d
        self.grid = jr.uniform(jr.key(seed), (n, d))

    def reference(self, ts: Float[Array, "m d"]) -> Float[Array, "m"]:
        """The curve a perfect candidate would reproduce, for plotting."""
        return jnp.sinc(2 * jnp.pi * ts - jnp.pi).mean(axis=-1)

    def __call__(self, f: Callable[[Float[Array, "d"]], Scalar]) -> Scalar:
        target = self.reference(self.grid)
        pred = jax.vmap(f)(self.grid)
        return jnp.mean(jnp.square(pred - target))


def bell_torque(t: Float[Array, "1"], center: float = 0.6, width: float = 0.12) -> Scalar:
    """A plausible assist profile: one push-off burst late in the gait cycle."""
    return jnp.exp(-(((t.squeeze() - center) / width) ** 2))


class ProfileMatch(FunctionalTestFunction):
    """Mean squared distance to a reference profile over the gait cycle.

    Unlike ``Ridge``, this depends on the candidate everywhere, not at a handful
    of probe points, so it is the honest test of whether the adaptive basis can
    actually shape a curve. Optimum is 0, at f == reference.
    """

    def __init__(self, reference=bell_torque, d: int = 1, n: int = 200):
        self.d = d
        self.reference_fn = reference
        self.grid = jnp.linspace(0.0, 1.0, n)[:, None]

    def reference(self, ts: Float[Array, "m d"]) -> Float[Array, "m"]:
        """The curve a perfect candidate would reproduce, for plotting."""
        return jax.vmap(self.reference_fn)(ts)

    def __call__(self, f: Callable[[Float[Array, "d"]], Scalar]) -> Scalar:
        target = self.reference(self.grid)
        pred = jax.vmap(f)(self.grid)
        return jnp.mean(jnp.square(pred - target))


class Ridge(FunctionalTestFunction):
    def __init__(self, profile: virtual_library.TestFunction, d: int, seed: int = 0):
        self.d = d
        self.profile = profile
        k1, k2, k3 = jr.split(jr.key(seed), 3)
        # sample d directions g = sum a * k(x, .)
        self.a = jr.uniform(k1, (d, d), minval=-1.0, maxval=1.0)
        self.x = jr.uniform(k2, (d, d, d), minval=0.0, maxval=1.0)
        # sample d biases b
        self.b = jr.uniform(k3, (d,), minval=-1.0, maxval=1.0)

    def __call__(self, f: Callable[[Float[Array, "d"]], Scalar]) -> Scalar:
        f = jax.vmap(jax.vmap(f))  # vectorize so it can be evaluated on x in one go
        g = self.b + jnp.sum(self.a * f(self.x), axis=-1)
        g = jax.nn.sigmoid(g)  # squash to [0, 1]
        return self.profile(g)
