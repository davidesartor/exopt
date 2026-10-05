"""bo-loop: propose a torque profile, run it on the controller, record it, repeat."""

import argparse
import time
import uuid
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr

from jaxtyping import Array, Float, Key
from exopt import acquisition, link, trials
from exopt.gaussian_process import GaussianProcess
from exopt.profiles import Profile


@partial(jax.jit, static_argnames=("harmonics", "raw_samples", "max_restarts"))
def propose_next(
    key: Key,
    x: Profile,
    y: Float[Array, "n"],
    harmonics: int = 1,
    amplitude: float = acquisition.AMPLITUDE,
    raw_samples: int = 256,
    max_restarts: int = 16,
) -> tuple[Profile, GaussianProcess]:
    """Return the next candidate Profile and the surrogate."""
    surrogate = GaussianProcess.fit(x, y)

    # score each box point as the peak-limited profile the controller will apply
    def loss(f: Profile):
        limited = acquisition.limit_peak(f, amplitude)
        return acquisition.acquisition_loss(surrogate, y.min(), limited)

    # maximize the acquisition from a screened space-filling start design
    lower, upper = acquisition.coefficient_bounds(harmonics, amplitude)
    unit = acquisition.latin_hypercube(key, 2 * harmonics, raw_samples)
    candidate = acquisition.optimize_restarts(
        loss,
        acquisition.box_points(unit, lower, upper),
        bounds=(lower, upper),
        max_restarts=max_restarts,
    )
    return acquisition.limit_peak(candidate, amplitude), surrogate


def initial_design(key: Key, harmonics: int, n: int) -> list[Profile]:
    """Latin hypercube of n profiles within the initial amplitude cap."""
    amplitude = acquisition.INITIAL_AMPLITUDE
    lower, upper = acquisition.coefficient_bounds(harmonics, amplitude)
    unit = acquisition.latin_hypercube(key, 2 * harmonics, n)
    design = acquisition.box_points(unit, lower, upper)
    return [
        acquisition.limit_peak(Profile(s, c), amplitude)
        for s, c in zip(design.sin, design.cos)
    ]


def next_profile(key: Key, history: list[trials.Trial], args) -> Profile:
    """Initial design until it is exhausted, then the BO proposal."""
    key_design, key_search = jr.split(key)
    if len(history) < args.n:
        return initial_design(key_design, args.harmonics, args.n)[len(history)]

    x = Profile.stack([t.profile for t in history])
    y = jnp.array([t.loss for t in history])
    candidate, _ = propose_next(
        jr.fold_in(key_search, len(history)),
        x.pad_to(max(x.harmonics, args.harmonics)),
        y,
        harmonics=args.harmonics,
        amplitude=acquisition.amplitude_cap(len(history)),
        raw_samples=args.raw_samples,
        max_restarts=args.max_restarts,
    )
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="controller host (tunnel endpoint)")
    parser.add_argument("--exp-dir", default="experiments/mock", help="folder recording the session")
    parser.add_argument("--duration", type=float, default=30.0, help="experiment time [min]")
    parser.add_argument("--harmonics", type=int, default=1, help="Fourier harmonics per profile")
    parser.add_argument("--n", type=int, default=4, help="initial design size")
    parser.add_argument("--warmup", type=int, default=100, help="samples dropped after a blend")
    parser.add_argument("--tol", type=float, default=0.05, help="relative standard error to stop a segment")
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-samples", type=int, default=1024)
    parser.add_argument("--max-restarts", type=int, default=16)
    args = parser.parse_args()

    samples, profiles = link.driver_link(args.host)
    deadline = time.monotonic() + 60.0 * args.duration
    while time.monotonic() < deadline:
        history = trials.load(args.exp_dir)
        profile = next_profile(jr.key(args.seed), history, args)

        # a fresh id per publish, so the controller never mistakes it for a stale profile
        index = max((t.index for t in history), default=0) + 1
        payload = profile.to_dict() | dict(id=f"{index}-{uuid.uuid4().hex[:6]}")
        collected = link.collect_segment(
            samples,
            profiles,
            payload,
            args.warmup,
            tol=args.tol,
            min_samples=args.min_samples,
            max_samples=args.max_samples,
        )
        trial = trials.save(args.exp_dir, index, profile, collected)
        best = min([t.loss for t in history] + [trial.loss])
        print(f"trial {index}: {len(collected)} samples, loss {trial.loss:.4f} (best {best:.4f})")


if __name__ == "__main__":
    main()
