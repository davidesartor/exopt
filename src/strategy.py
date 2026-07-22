"""Acquisition strategies: what to try next, given what has been observed.

Two modes share the same GP + log-EI machinery:

``propose_next``
    Classic BO over a fixed-dimension parameter vector in the unit cube.

``propose_next_functional``
    BO over torque profiles, represented as sparse RKHS functions. The
    acquisition optimizes the k basis locations, their values, *and* the
    candidate's own lengthscale rho jointly, so both where the profile needs
    resolution and how smooth it should be are chosen by the search rather than
    fixed up front.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar

from . import acquisition, designs, gp, rkhs

jax.config.update("jax_enable_x64", True)


def propose_next(
    xs: Float[Array, "n d"],
    ys: Float[Array, "n"],
    seed: int,
    acquisition_raw_samples: int = 256,
    acquisition_max_restarts: int = 5,
    domain: tuple[float, float] = designs.VECTOR_DOMAIN,
) -> tuple[Float[Array, "d"], gp.GaussianProcess]:
    """Maximize log-EI over ``domain``, from a Latin hypercube of restarts."""
    dim = xs.shape[-1]
    surrogate_model = gp.GaussianProcess().fit(xs, ys)

    @jax.jit
    @jax.value_and_grad
    def acquisition_loss(x: Float[Array, "d"]) -> Scalar:
        mu, cov = surrogate_model.predict(x[None, :])
        return -acquisition.log_expected_improvement(
            mu=mu.squeeze(),
            sigma=cov.squeeze() ** 0.5,
            y_best=surrogate_model.observed_ys.min(),
        )

    candidates = designs.latin_hypercube(dim, acquisition_raw_samples, seed, domain)
    x_next = acquisition.optimize_restarts(
        acquisition_loss=acquisition_loss,
        candidates=candidates,
        max_restarts=acquisition_max_restarts,
        domain=domain,
    )
    return x_next, surrogate_model


def propose_next_functional(
    fs: list[rkhs.Function],
    ys: Float[Array, "n"],
    k: int,
    seed: int,
    d: int = 1,
    y_range: tuple[float, float] = (-1.0, 1.0),
    acquisition_raw_samples: int = 256,
    acquisition_max_restarts: int = 5,
    rho: Float[Array, "d"] | None = None,
) -> tuple[rkhs.Function, gp.FunctionalGaussianProcess]:
    """Maximize log-EI over sparse RKHS functions with an adaptive lengthscale.

    The search variable is a flat vector in [0,1]^(k*(d+1)+d): per basis point,
    d coordinates plus one value, then d trailing entries for the candidate's
    own lengthscale rho (searched in log scale over ``rkhs.RHO_RANGE``). So the
    smoothness of the profile is chosen by the acquisition rather than fixed up
    front and swept offline.

    Passing ``rho`` pins the lengthscale instead of searching it, dropping the
    trailing d entries -- the fixed-lengthscale baseline this method is meant to
    improve on.
    """
    ambient = rkhs.ambient_space(d)
    surrogate_model = gp.FunctionalGaussianProcess(ambient=ambient).fit(fs, ys)
    dim = k * (d + 1) + d if rho is None else k * (d + 1)

    def decode(p: Float[Array, "dim"]) -> rkhs.Function:
        # from_array only reshapes for itself when it has to peel rho off first
        if rho is None:
            return rkhs.Function.from_array(None, p, d=d, y_range=y_range)
        return rkhs.Function.from_array(rho, p.reshape(k, d + 1), y_range=y_range)

    @jax.jit
    @jax.value_and_grad
    def acquisition_loss(p: Float[Array, "k*(d+1)+d"]) -> Scalar:
        mu, cov = surrogate_model.predict([decode(p)])
        return -acquisition.log_expected_improvement(
            mu=mu.squeeze(),
            sigma=cov.squeeze() ** 0.5,
            y_best=surrogate_model.observed_ys.min(),
        )

    # value only, so the log-EI branches never need to stay gradient-safe
    @jax.jit
    def screening_loss(ps: Float[Array, "n k*(d+1)+d"]) -> Float[Array, "n"]:
        mu, cov = surrogate_model.predict_marginals(jax.vmap(decode)(ps))
        return -jax.vmap(acquisition.log_expected_improvement)(
            mu.squeeze(-1),
            cov.squeeze((-2, -1)) ** 0.5,
            jnp.full(len(ps), surrogate_model.observed_ys.min()),
        )

    candidates = designs.latin_hypercube(dim, acquisition_raw_samples, seed)
    p_next = acquisition.optimize_restarts(
        acquisition_loss=acquisition_loss,
        candidates=candidates,
        max_restarts=acquisition_max_restarts,
        screening_loss=screening_loss,
    )
    return decode(p_next), surrogate_model


def initial_functions(
    k: int,
    n: int,
    seed: int,
    d: int = 1,
    y_range: tuple[float, float] = (-1.0, 1.0),
    edges: bool = False,
    rho: Float[Array, "d"] | None = None,
) -> list[rkhs.Function]:
    """Initial design in function space: space-filling over (x, y) and rho.

    Passing ``rho`` pins the lengthscale, matching the fixed-lengthscale
    baseline so both start from designs of the same size.
    """
    sampler = designs.edge_prioritized if edges else designs.latin_hypercube
    dim = k * (d + 1) + d if rho is None else k * (d + 1)
    ps = sampler(dim, n, seed)
    if rho is None:
        return [rkhs.Function.from_array(None, p, d=d, y_range=y_range) for p in ps]
    return [
        rkhs.Function.from_array(rho, p.reshape(k, d + 1), y_range=y_range) for p in ps
    ]
