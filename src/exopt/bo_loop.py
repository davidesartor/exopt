"""BO proposal and the online optimization loop."""

import argparse
import os
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr

from jaxtyping import Array, Float, Key
from exopt import acquisition, experiment_io, zmq_link
from exopt.gaussian_process import GaussianProcess
from exopt.rkhs_functions import Profile


@partial(jax.jit, static_argnames=("harmonics", "raw_samples", "max_restarts"))
def propose_next(
    key: Key,
    x: Profile,
    y: Float[Array, "n"],
    harmonics: int = 1,
    raw_samples: int = 256,
    max_restarts: int = 16,
) -> tuple[Profile, GaussianProcess]:
    """Return the next candidate Profile and the surrogate."""
    # fit the surrogate on RKHS distances between observed profiles
    surrogate = GaussianProcess.fit(x, y)
    loss = partial(acquisition.acquisition_loss, surrogate, y.min())

    # maximize the acquisition from a screened space-filling start design
    lower, upper = acquisition.coefficient_bounds(harmonics)
    unit = acquisition.latin_hypercube(key, 2 * harmonics, raw_samples)
    starts = jax.tree.map(
        lambda lo, hi, u: lo + u * (hi - lo),
        lower, upper, Profile(unit[:, :harmonics], unit[:, harmonics:]),
    )
    candidate = acquisition.optimize_restarts(
        loss, starts, bounds=(lower, upper), max_restarts=max_restarts
    )
    return candidate, surrogate


def load_dataset(exp_dir: str) -> tuple[Profile, Float[Array, "n"]]:
    """All completed trials as one stacked Profile (padded to the largest H)."""
    completed = sorted(
        experiment_io.config_numbers(exp_dir) & experiment_io.run_numbers(exp_dir)
    )
    profiles = [
        zmq_link.config_torque_profile(experiment_io.read_config(exp_dir, i))
        for i in completed
    ]
    ys = jnp.array([experiment_io.read_result(exp_dir, i) for i in completed])

    # pad every trial to a shared harmonic count and stack into one Profile
    H = max((p.harmonics for p in profiles), default=1)
    profiles = [p.pad_to(H) for p in profiles]
    stacked = Profile(
        sin=jnp.stack([p.sin for p in profiles]),
        cos=jnp.stack([p.cos for p in profiles]),
    )
    return stacked, ys


def seed_folder(args) -> None:
    """Fill an empty folder with a Latin hypercube design of configs."""
    harmonics = args.harmonics if args.mode == "functional" else 1
    lower, upper = acquisition.coefficient_bounds(harmonics)
    unit = acquisition.latin_hypercube(jr.key(args.seed), 2 * harmonics, args.n)
    for offset, u in enumerate(unit):
        p = jax.tree.map(
            lambda lo, hi, uu: lo + uu * (hi - lo),
            lower, upper, Profile(u[:harmonics], u[harmonics:]),
        )
        payload = zmq_link.profile_payload(p)
        experiment_io.save_config(args.exp_dir, 1 + offset, payload)
        print(f"config_{1 + offset}.json: {payload}")


def load_with_fantasies(exp_dir: str):
    """Completed runs, plus surrogate-mean fantasy values for still-queued configs."""
    observed, ys = load_dataset(exp_dir)
    queued = sorted(
        experiment_io.config_numbers(exp_dir) - experiment_io.run_numbers(exp_dir)
    )
    if not queued:
        return observed, ys

    # pad completed and queued trials to a shared harmonic count
    profiles = [
        zmq_link.config_torque_profile(experiment_io.read_config(exp_dir, i))
        for i in queued
    ]
    H = max(observed.harmonics, max(p.harmonics for p in profiles))
    observed = observed.pad_to(H)
    profiles = [p.pad_to(H) for p in profiles]

    # fantasize each queued config at the surrogate posterior mean
    surrogate = GaussianProcess.fit(observed, ys)
    fantasies = jnp.array([surrogate.predict(p)[0].squeeze() for p in profiles])
    stacked = Profile(
        sin=jnp.concatenate([observed.sin, jnp.stack([p.sin for p in profiles])]),
        cos=jnp.concatenate([observed.cos, jnp.stack([p.cos for p in profiles])]),
    )
    return stacked, jnp.concatenate([ys, fantasies])


def propose(args, key: Key) -> None:
    """One BO step: load the folder, maximize the acquisition, save the config."""
    # auto mode continues whatever the newest config in the folder used
    mode = args.mode
    if mode == "auto":
        mode = experiment_io.newest_mode(args.exp_dir)

    observed, ys = load_with_fantasies(args.exp_dir)
    sign = 1.0 if args.minimize else -1.0
    print(
        f"Loaded {len(ys)} observations (incl. fantasies), "
        f"best so far: {(sign * ys).min():.6f}"
    )

    # a vector candidate is a single fundamental harmonic
    harmonics = 1 if mode == "vector" else args.harmonics
    # pad the history only if the new candidate space is wider
    observed = observed.pad_to(max(observed.harmonics, harmonics))
    candidate, _ = propose_next(
        key,
        observed,
        sign * ys,
        harmonics=harmonics,
        raw_samples=args.raw_samples,
        max_restarts=args.max_restarts,
    )

    i = experiment_io.next_index(args.exp_dir)
    payload = zmq_link.profile_payload(candidate)
    experiment_io.save_config(args.exp_dir, i, payload)
    print(f"Proposed config_{i}.json: {payload}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", default="localhost", help="controller host (tunnel endpoint)"
    )
    parser.add_argument(
        "--exp-dir",
        default="experiments/mock-local",
        help="folder recording the session",
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="total segments to run"
    )
    parser.add_argument(
        "--warmup", type=int, default=100, help="samples dropped after a swap"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.05,
        help="relative standard-error tolerance for segment convergence",
    )
    parser.add_argument(
        "--min-samples", type=int, default=100, help="samples before checking convergence"
    )
    parser.add_argument(
        "--max-samples", type=int, default=2000, help="per-segment sample budget"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "vector", "functional"],
        default="auto",
        help="auto continues the newest config's mode (vector on an empty folder); "
        "functional can extend a vector history",
    )
    parser.add_argument("--n", type=int, default=4, help="initial design size")
    parser.add_argument("--harmonics", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-samples", type=int, default=1024)
    parser.add_argument("--max-restarts", type=int, default=16)
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="minimize the objective (default: maximize)",
    )
    args = parser.parse_args()

    # connect to the controller and resume (or seed) the experiment folder
    key = jr.key(args.seed)
    os.makedirs(args.exp_dir, exist_ok=True)
    samples, profiles = zmq_link.driver_link(args.host)
    if not experiment_io.config_numbers(args.exp_dir):
        print(f"Empty folder: seeding a {args.n}-point Latin hypercube design.")
        seed_folder(args)

    while True:
        # split the folder into completed and still-pending configs
        configs = experiment_io.config_numbers(args.exp_dir)
        done = configs & experiment_io.run_numbers(args.exp_dir)
        pending = sorted(configs - done)
        if not pending:
            if len(done) >= args.iterations:
                break
            key, subkey = jr.split(key)
            propose(args, subkey)
            continue

        # signal the controller to start on the head of the queue
        i = pending[0]
        payload = experiment_io.read_config(args.exp_dir, i) | dict(id=i)
        profiles.send_json(payload)

        # while the controller runs it, extend the queue against fantasized outcomes
        if len(done) >= 2 and len(configs) < args.iterations:
            key, subkey = jr.split(key)
            propose(args, subkey)

        # gather the segment and record it next to its config
        collected = zmq_link.collect_segment(
            samples,
            profiles,
            payload,
            args.warmup,
            tol=args.tol,
            min_samples=args.min_samples,
            max_samples=args.max_samples,
        )
        experiment_io.write_run(args.exp_dir, i, collected)
        print(
            f"segment {i}: {len(collected)} samples, loss = "
            f"{experiment_io.read_result(args.exp_dir, i):.6f}"
        )

    print(f"Done: {args.iterations} segments in {args.exp_dir}.")


if __name__ == "__main__":
    main()
