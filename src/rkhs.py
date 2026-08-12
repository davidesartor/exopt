
from typing import ClassVar, NamedTuple, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar

from . import kernels

jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))

RHO_RANGE = (0.05, 0.4)


class Function(NamedTuple):
    rho: Float[Array, "d"]
    x: Float[Array, "k d"]
    a: Float[Array, "k"]

    @staticmethod
    def stack(fs: list["Function"]) -> "Function":
        k = max(len(f.x) for f in fs)
        fs = [f if len(f.x) == k else f.with_basis_size(k) for f in fs]
        return jax.tree.map(lambda *leaves: jnp.stack(leaves), *fs)

    @eqx.filter_jit
    def __call__(self, t: Float[Array, "d"]) -> Scalar:
        Ktx = RKHS(self.rho)(t[None, :], self.x)
        return (Ktx @ self.a).squeeze()

    @eqx.filter_jit
    def sample(self, ts: Float[Array, "m d"]) -> Float[Array, "m"]:
        return RKHS(self.rho)(ts, self.x) @ self.a

    @classmethod
    def from_array(
        cls,
        rho: Float[Array, "d"] | None,
        p: Float[Array, "k*(d+1)+d"],
        d: int | None = None,
        rho_range: tuple[float, float] = RHO_RANGE,
        x_range: tuple[float, float] = (0.0, 1.0),
        y_range: tuple[float, float] = (-1.0, 1.0),
        eps: float = 0.01,
    ) -> Self:
        if rho is None:
            rho, p = cls.split_rho(p, d, rho_range)
            p = p.reshape(-1, d + 1)

        x, y = p[:, :-1], p[:, -1]
        x = x * (x_range[1] - x_range[0]) + x_range[0]
        y = y * (y_range[1] - y_range[0]) + y_range[0]
        return cls.from_xy(rho, x, y, eps=eps)

    @classmethod
    def from_xy(
        cls,
        rho: Float[Array, "d"] | None,
        x: Float[Array, "k d"],
        y: Float[Array, "k"],
        rho_range: tuple[float, float] = RHO_RANGE,
        eps: float = 0.01,
    ) -> Self:
        if rho is None:
            rho, y = cls.split_rho(y, x.shape[-1], rho_range)

        Kxx = RKHS(rho)(x, x) + eps * jnp.eye(len(x))
        a = jnp.linalg.solve(Kxx, y)
        return cls(rho=rho, a=a, x=x)

    @eqx.filter_jit
    def with_basis_size(self, k: int) -> Self:
        assert k >= len(self.x), "Can only pad the basis up"
        padding = k - len(self.x)
        return self._replace(
            x=jnp.pad(self.x, ((0, padding), (0, 0))),
            a=jnp.pad(self.a, (0, padding)),
        )

    @staticmethod
    def split_rho(
        p: Float[Array, "n+d"],
        d: int,
        rho_range: tuple[float, float] = RHO_RANGE,
    ) -> tuple[Float[Array, "d"], Float[Array, "n"]]:
        log_min, log_max = jnp.log(rho_range[0]), jnp.log(rho_range[1])
        log_rho = p[-d:] * (log_max - log_min) + log_min
        return jnp.exp(log_rho), p[:-d]


class RKHS(eqx.Module):

    rho: Float[Array, "d"]
    metric: ClassVar[kernels.Metric] = kernels.Euclidean()
    profile: ClassVar[kernels.Profile] = kernels.SquaredExponential()

    @property
    def d(self) -> int:
        return len(self.rho)

    @eqx.filter_jit
    def __call__(
        self,
        xs1: Float[Array, "n d"],
        xs2: Float[Array, "m d"],
    ) -> Float[Array, "n m"]:
        return self.profile(self.metric(self.rho, xs1, xs2))

    @eqx.filter_jit
    def inner(self, f1: Function, f2: Function) -> Scalar:
        l0, l1, l2 = self.rho**2, f1.rho**2, f2.rho**2
        ls = l1 + l2 - l0

        scale = jnp.sqrt(jnp.prod(l1 * l2 / (l0 * ls)))

        return scale * (f1.a @ RKHS(jnp.sqrt(ls))(f1.x, f2.x) @ f2.a)

    @eqx.filter_jit
    def distance(self, f1: Function, f2: Function) -> Scalar:
        d2 = self.inner(f1, f1) + self.inner(f2, f2) - 2 * self.inner(f1, f2)
        positive = d2 > 0.0
        return jnp.where(positive, jnp.sqrt(jnp.where(positive, d2, 1.0)), 0.0)

    @eqx.filter_jit
    def pairwise_distances(self, fs1: Function, fs2: Function) -> Float[Array, "m n"]:
        return jax.vmap(jax.vmap(self.distance, (None, 0)), (0, None))(fs1, fs2)


def ambient_space(d: int = 1, rho_range: tuple[float, float] = RHO_RANGE) -> RKHS:
    return RKHS(rho=jnp.full(d, rho_range[0]))
