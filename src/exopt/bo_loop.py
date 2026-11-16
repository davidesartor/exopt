"""bo-loop: propose a torque profile, run it on the controller, record it, repeat."""

from functools import partial
from jaxtyping import Array, Float, Key

import jax
import jax.numpy as jnp
import jax.random as jr

import argparse
import os
import signal
import uuid
from pathlib import Path

from einops import rearrange
from bofus import acquisition
from bofus.gp import GaussianProcess
from bofus.rkhs import X_MAX, RBFMixture
from bofus.utils import latin_hypercube_sample, rescale

from exopt import link, trials
from exopt.controller import MAX_TORQUE, PERIODIC
from exopt.review import write_review

COMPILATION_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "jax"  # compiled proposals, one per history capacity


def variance_range(settings: trials.Settings) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Atom variances l (domain [-X_MAX, X_MAX] for the cycle) of the settings' lengthscales in rad of gait phase."""
    to_variance = lambda lengthscale: jnp.asarray((lengthscale * X_MAX / jnp.pi) ** 2)
    return to_variance(settings.min_lengthscale), to_variance(settings.max_lengthscale)


def design(key: Key, atoms: int, n: int, l_range: tuple[Float[Array, ""], Float[Array, ""]]) -> RBFMixture:
    """Latin hypercube of n profiles over the acquisition's box, stacked."""
    u = latin_hypercube_sample(key, (n, atoms, 3))
    log_l = rescale(u[..., 0], *map(jnp.log, l_range))
    x = rescale(u[..., 1], -X_MAX, X_MAX)
    a = rescale(u[..., 2], -acquisition.A_MAX, acquisition.A_MAX)
    return RBFMixture(jnp.exp(log_l)[..., None], x[..., None], a)


def stack(
    profiles: list[RBFMixture], losses: list[float], ses: list[float]
) -> tuple[RBFMixture, Float[Array, "n"], Float[Array, "n"]]:
    """One batched history, nan-padded to a power-of-2 length so proposals recompile only when it doubles."""
    m = max(p.a.shape[-1] for p in profiles)
    n = max(8, 1 << (len(profiles) - 1).bit_length())
    padded = [p.pad_to(m) for p in profiles]
    padded += [padded[0]] * (n - len(padded))
    ys = jnp.full(n, jnp.nan).at[: len(losses)].set(jnp.array(losses))
    # an undefined SE counts as noiseless, the fitted nugget still applies
    y_stds = jnp.zeros(n).at[: len(ses)].set(jnp.nan_to_num(jnp.array(ses), nan=0.0, posinf=0.0))
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *padded), ys, y_stds


def best_posterior(surrogate: GaussianProcess) -> dict[str, Float[Array, ""]]:
    """Posterior mean and sd of the observed profile the surrogate believes best, and the mean posterior sd over all."""
    mean, var = surrogate.predict_marginal(surrogate.f)
    sd = jnp.sqrt(var)
    observed = jnp.isfinite(surrogate.y)
    best = jnp.where(observed, mean, jnp.inf).argmin()
    return dict(best_mean=mean[best], best_sd=sd[best], mean_sd=jnp.average(sd, weights=observed))


@partial(jax.jit, static_argnames=("atoms", "raw_samples", "max_restarts"))
def propose_next(
    key: Key,
    history: RBFMixture,
    ys: Float[Array, "n"],
    y_stds: Float[Array, "n"],
    l_range: tuple[Float[Array, ""], Float[Array, ""]],
    atoms: int = 2,
    raw_samples: int = 1024,
    max_restarts: int = 32,
) -> tuple[RBFMixture, dict[str, Float[Array, ""]]]:
    """Maximize log-EI over profiles of at least `atoms` atoms; also the posterior summary the stop rule reads."""
    key_fit, key_acquisition = jr.split(key)

    # one output per profile, padded so candidates get `atoms` atoms
    history = history.pad_to(max(atoms, history.a.shape[-1]))
    f = jax.tree.map(lambda z: rearrange(z, "n ... -> n 1 ..."), history)
    surrogate = GaussianProcess.fit(key_fit, f, ys, y_std=y_stds, periodic=PERIODIC)

    candidate = acquisition.optimize_expected_improvement(
        key_acquisition, surrogate, l_range, multi_starts=max_restarts, n_probes=raw_samples
    )
    return jax.tree.map(lambda z: z[0, 0], candidate), best_posterior(surrogate)


def enable_compilation_cache() -> None:
    jax.config.update("jax_compilation_cache_dir", str(COMPILATION_CACHE))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)


def next_profile(key: Key, history: list[trials.Trial], settings: trials.Settings) -> tuple[RBFMixture, dict | None]:
    """Initial design until it is exhausted, then the BO proposal with the best posterior."""
    key_design, key_search = jr.split(key)
    if len(history) < settings.n:
        initial = design(key_design, settings.atoms, settings.n, variance_range(settings))
        return jax.tree.map(lambda a: a[len(history)], initial), None

    candidate, best = propose_next(
        jr.fold_in(key_search, len(history)),
        *stack([t.profile for t in history], [t.loss for t in history], [t.se for t in history]),
        variance_range(settings),
        atoms=settings.atoms,
        raw_samples=settings.raw_samples,
        max_restarts=settings.max_restarts,
    )
    return candidate, {k: float(v) for k, v in best.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="controller host (tunnel endpoint)")
    parser.add_argument("--exp-dir", default="experiments/mock", help="folder recording the session")
    parser.add_argument("--duration", type=float, default=30.0, help="experiment time [min]")
    parser.add_argument("--atoms", type=int, default=2, help="periodized RBF atoms per profile")
    parser.add_argument("--mode", choices=EMBEDDINGS, default="functional",
                        help="GP on the L2 distance between profiles, or on a single atom's coefficients")
    parser.add_argument("--n", type=int, default=4, help="initial design size")
    parser.add_argument("--warmup", type=int, default=100, help="samples dropped after the ramp up")
    parser.add_argument("--z", type=float, default=3.0, help="stop a segment this many sd above the best")
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-samples", type=int, default=1024)
    parser.add_argument("--max-restarts", type=int, default=16)
    parser.add_argument("--warm-cache", type=int, metavar="N",
                        help="only compile the proposals for histories up to N trials, then exit")
    args = parser.parse_args()
    if args.mode == "coefficients":
        args.atoms = 1

    enable_compilation_cache()

    samples, profiles = link.driver_link(args.host)
    try:
        run_session(samples, profiles, deadline, args)
    finally:
        link.release(profiles)


def run_session(samples, profiles, deadline: float, args) -> None:
    """Propose, ramp up, collect, release, record, until the deadline."""
    while time.monotonic() < deadline:
        history = trials.load(args.exp_dir)
        profile, best = next_profile(jr.key(args.seed), history, args)

        # a fresh id per publish, so the controller never mistakes it for a stale profile
        index = max((t.index for t in history), default=0) + 1
        payload = profile.to_dict() | dict(id=f"{index}-{uuid.uuid4().hex[:6]}")
        if best is not None:
            payload["stop_rule"] = best | dict(z=args.z)
        trials.save_current(args.exp_dir, payload)
        collected = link.collect_segment(
            samples, profiles, payload, args.warmup,
            min_samples=args.min_samples, max_samples=args.max_samples,
        )
        # ramp down to zero while the next proposal is computed
        link.release(profiles)
        trial = trials.save(args.exp_dir, index, profile, collected)
        best = min([t.loss for t in history] + [trial.loss])
        print(f"trial {index}: {len(collected)} samples, loss {trial.loss:.4f} ± {trial.se:.4f} (best {best:.4f})")


if __name__ == "__main__":
    main()
