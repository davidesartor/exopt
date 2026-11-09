"""bo-loop: propose a torque profile, run it on the controller, record it, repeat."""

from functools import partial
from jaxtyping import Array, Float, Key, Scalar

import jax
import jax.numpy as jnp
import jax.random as jr

import argparse
import os
import signal
import time
import uuid
from pathlib import Path

from exopt import acquisition, link, trials
from exopt.gaussian_process import GaussianProcess
from exopt.profiles import EMBEDDINGS, Profile

COMPILATION_CACHE = Path.home() / ".cache" / "exopt" / "jax"  # compiled proposals, one per history size


def design(key: Key, atoms: int, n: int) -> Profile:
    """Latin hypercube of n profiles, stacked."""
    lower, upper = acquisition.bounds(atoms)
    return acquisition.latin_hypercube(key, lower, upper, n)


def best_posterior(surrogate: GaussianProcess) -> dict[str, Float[Array, ""]]:
    """Posterior mean and sd of the observed profile the surrogate believes best, and the mean posterior sd over all."""
    mean, cov = surrogate.predict(surrogate.xs)
    sd = jnp.sqrt(jnp.diag(cov))
    return dict(best_mean=mean.min(), best_sd=sd[mean.argmin()], mean_sd=sd.mean())


@partial(jax.jit, static_argnames=("atoms", "mode", "raw_samples", "max_restarts"))
def propose_next(
    key: Key,
    xs: Profile,
    ys: Float[Array, "n"],
    ses: Float[Array, "n"],
    atoms: int = 2,
    mode: str = "functional",
    raw_samples: int = 256,
    max_restarts: int = 16,
) -> tuple[Profile, dict[str, Float[Array, ""]]]:
    """Maximize log-EI over profiles of `atoms` atoms; also the posterior summary the stop rule reads."""
    embed = EMBEDDINGS[mode]
    surrogate = GaussianProcess.fit(embed(xs), ys, ses**2)
    best = best_posterior(surrogate)

    # improvement over the lowest posterior mean among observed profiles (measurements are noisy)
    def negative_log_ei(f: Profile) -> Scalar:
        mu, cov = surrogate.predict(embed(f))
        return -acquisition.log_expected_improvement(mu.squeeze(), cov.squeeze() ** 0.5, best["best_mean"])

    lower, upper = acquisition.bounds(atoms)
    candidate = acquisition.optimize_restarts(
        negative_log_ei,
        acquisition.latin_hypercube(key, lower, upper, raw_samples),
        bounds=(lower, upper),
        max_restarts=max_restarts,
    )
    return candidate, best


def enable_compilation_cache() -> None:
    jax.config.update("jax_compilation_cache_dir", str(COMPILATION_CACHE))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)


def compile_proposal(n: int, atoms: int, mode: str, raw_samples: int, max_restarts: int) -> int:
    """Compile (and cache) the proposal for a history of n trials, as next_profile calls it."""
    xs = design(jr.key(n), atoms, n)
    propose_next.lower(
        jr.key(0), xs, jnp.zeros(n), jnp.ones(n), atoms=atoms, mode=mode,
        raw_samples=raw_samples, max_restarts=max_restarts,
    ).compile()
    return n


def warm_cache(args, up_to: int) -> None:
    """Compile the proposal for every history size up to `up_to` (any initial design size)."""
    # serial: XLA already compiles on all cores, so threads or processes gain nothing
    start = time.monotonic()
    for n in range(1, up_to + 1):
        compile_proposal(n, args.atoms, args.mode, args.raw_samples, args.max_restarts)
        print(f"compiled history size {n} ({time.monotonic() - start:.0f} s)", flush=True)


def next_profile(key: Key, history: list[trials.Trial], args) -> tuple[Profile, dict | None]:
    """Initial design until it is exhausted, then the BO proposal with the best posterior."""
    key_design, key_search = jr.split(key)
    if len(history) < args.n:
        initial = design(key_design, args.atoms, args.n)
        return jax.tree.map(lambda a: a[len(history)], initial), None

    candidate, best = propose_next(
        jr.fold_in(key_search, len(history)),
        Profile.stack([t.profile for t in history]),
        jnp.array([t.loss for t in history]),
        jnp.array([t.se for t in history]),
        atoms=args.atoms,
        mode=args.mode,
        raw_samples=args.raw_samples,
        max_restarts=args.max_restarts,
    )
    return candidate, {k: float(v) for k, v in best.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="controller host (tunnel endpoint)")
    parser.add_argument("--exp-dir", default="experiments/mock", help="folder recording the session")
    parser.add_argument("--atoms", type=int, default=2, help="periodized RBF atoms per profile")
    parser.add_argument("--mode", choices=EMBEDDINGS, default="functional",
                        help="GP on the L2 distance between profiles, or on a single atom's coefficients")
    parser.add_argument("--n", type=int, default=4, help="initial design size")
    parser.add_argument("--min-gaits", type=int, default=5, help="valid gaits a profile runs at least (2 or more)")
    parser.add_argument("--max-gaits", type=int, default=20, help="valid gaits a profile runs at most")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-samples", type=int, default=1024)
    parser.add_argument("--max-restarts", type=int, default=16)
    parser.add_argument("--warm-cache", type=int, metavar="N",
                        help="only compile the proposals for histories up to N trials, then exit")
    args = parser.parse_args()
    if args.mode == "coefficients":
        args.atoms = 1
    # an SE needs 2 gaits
    args.min_gaits = max(args.min_gaits, 2)
    args.max_gaits = max(args.max_gaits, args.min_gaits)

    # SIGINT stops the session cleanly even when launched with it ignored (a background job)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    if os.name == "nt":  # the dashboard's stop on Windows is a ctrl-break
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    enable_compilation_cache()
    if args.warm_cache:
        warm_cache(args, args.warm_cache)
        return

    samples, profiles = link.driver_link(args.host)
    try:
        run_session(samples, profiles, args)
    except ConnectionError as error:
        print(f"Stopped: {error}")
    finally:
        link.release(profiles)
        from exopt.dashboard import write_review
        print(f"Review in {write_review(args.exp_dir, args.mode)}")


def run_session(samples, profiles, args) -> None:
    """Propose, ramp up, collect, release, record, until interrupted or the controller is gone."""
    while True:
        history = trials.load(args.exp_dir)
        profile, best = next_profile(jr.key(args.seed), history, args)

        # a fresh id per publish, so the controller never mistakes it for a stale profile
        index = max((t.index for t in history), default=0) + 1
        payload = profile.to_dict() | dict(id=f"{index}-{uuid.uuid4().hex[:6]}")
        payload["stop_rule"] = dict(min_gaits=args.min_gaits, max_gaits=args.max_gaits) | (best or {})
        trials.save_current(args.exp_dir, payload)
        collected, reason = link.collect_segment(samples, profiles, payload)
        # ramp down to zero while the next proposal is computed
        link.release(profiles)
        trial = trials.save(args.exp_dir, index, profile, collected)
        best = min([t.loss for t in history] + [trial.loss])
        print(f"trial {index}: {len(trials.gait_slices(trial.samples))} valid gaits ({reason}), loss {trial.loss:.4f} ± {trial.se:.4f} (best {best:.4f})")


if __name__ == "__main__":
    main()
