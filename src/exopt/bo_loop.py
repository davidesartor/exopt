"""BO proposal and the online optimization loop."""

import argparse
import os

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar

from exopt import acquisition, experiment_io, rkhs_functions, zmq_link
from exopt.gaussian_process import GaussianProcess
from exopt.rkhs_functions import Profile


def propose_next(
    observed: Profile,
    ys: Float[Array, "n"],
    k: int,
    harmonics: int,
    seed: int,
    raw_samples: int = 256,
    max_restarts: int = 5,
) -> tuple[Float[Array, "k"], Float[Array, "k"], GaussianProcess]:
    """Return (amplitudes, phases) of the next k-atom candidate and the surrogate."""
    H = max(observed.harmonics, harmonics)
    observed = observed.pad_to(H)

    dists = rkhs_functions.pairwise_distances(observed, observed)
    surrogate = GaussianProcess.fit(dists, ys)
    y_best = ys.min()

    def loss(p: Float[Array, "2k"]) -> Scalar:
        f = rkhs_functions.from_vector(p, harmonics).pad_to(H)
        dists_ox = jax.vmap(rkhs_functions.distance, in_axes=(0, None))(observed, f)
        mu, cov = surrogate.predict(dists_ox)
        return -acquisition.log_expected_improvement(
            mu=mu.squeeze(), sigma=cov.squeeze() ** 0.5, y_best=y_best
        )

    lower, upper = rkhs_functions.vector_bounds(k)
    unit = acquisition.latin_hypercube(2 * k, raw_samples, seed)
    starts = lower + unit * (upper - lower)
    p_next = acquisition.optimize_restarts(
        loss, starts, bounds=(lower, upper), max_restarts=max_restarts
    )
    return p_next[:k], p_next[k:], surrogate


# --- online loop ----------------------------------------------------------


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

    H = max((p.harmonics for p in profiles), default=1)
    profiles = [p.pad_to(H) for p in profiles]
    stacked = Profile(
        sin=jnp.stack([p.sin for p in profiles]),
        cos=jnp.stack([p.cos for p in profiles]),
    )
    return stacked, ys


def seed_folder(args) -> None:
    k, harmonics = (args.k, args.harmonics) if args.mode == "functional" else (1, 1)
    lower, upper = rkhs_functions.vector_bounds(k)
    unit = acquisition.latin_hypercube(2 * k, args.n, args.seed)
    for offset, u in enumerate(unit):
        p = lower + u * (upper - lower)
        amplitude, phase = p[:k], p[k:]
        payload = zmq_link.profile_payload(amplitude, phase, harmonics)
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

    profiles = [
        zmq_link.config_torque_profile(experiment_io.read_config(exp_dir, i))
        for i in queued
    ]
    H = max(observed.harmonics, max(p.harmonics for p in profiles))
    observed = observed.pad_to(H)
    profiles = [p.pad_to(H) for p in profiles]

    surrogate = GaussianProcess.fit(rkhs_functions.pairwise_distances(observed, observed), ys)
    fantasies = jnp.array([
        surrogate.predict(jax.vmap(rkhs_functions.distance, in_axes=(0, None))(observed, p))
        .mean.squeeze()
        for p in profiles
    ])
    stacked = Profile(
        sin=jnp.concatenate([observed.sin, jnp.stack([p.sin for p in profiles])]),
        cos=jnp.concatenate([observed.cos, jnp.stack([p.cos for p in profiles])]),
    )
    return stacked, jnp.concatenate([ys, fantasies])


def propose(args) -> None:
    mode = args.mode
    if mode == "auto":
        mode = experiment_io.newest_mode(args.exp_dir)

    observed, ys = load_with_fantasies(args.exp_dir)
    sign = 1.0 if args.minimize else -1.0
    print(f"Loaded {len(ys)} observations (incl. fantasies), "
          f"best so far: {(sign * ys).min():.6f}")

    k, harmonics = (1, 1) if mode == "vector" else (args.k, args.harmonics)
    amplitude, phase, _ = propose_next(
        observed,
        sign * ys,
        k=k,
        harmonics=harmonics,
        seed=args.seed,
        raw_samples=args.raw_samples,
        max_restarts=args.max_restarts,
    )

    i = experiment_io.next_index(args.exp_dir)
    payload = zmq_link.profile_payload(amplitude, phase, harmonics)
    experiment_io.save_config(args.exp_dir, i, payload)
    print(f"Proposed config_{i}.json: {payload}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="controller host (tunnel endpoint)")
    parser.add_argument(
        "--exp-dir", default="experiments/mock-local", help="folder recording the session"
    )
    parser.add_argument("--iterations", type=int, default=10, help="total segments to run")
    parser.add_argument("--samples", type=int, default=1000, help="samples per segment")
    parser.add_argument("--warmup", type=int, default=100, help="samples dropped after a swap")
    parser.add_argument(
        "--mode", choices=["auto", "vector", "functional"], default="auto",
        help="auto continues the newest config's mode (vector on an empty folder); "
        "functional can extend a vector history",
    )
    parser.add_argument("--n", type=int, default=4, help="initial design size")
    parser.add_argument("--k", type=int, default=3, help="atoms per functional config")
    parser.add_argument("--harmonics", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-samples", type=int, default=1024)
    parser.add_argument("--max-restarts", type=int, default=16)
    parser.add_argument(
        "--minimize", action="store_true",
        help="minimize the objective (default: maximize)",
    )
    args = parser.parse_args()

    os.makedirs(args.exp_dir, exist_ok=True)
    samples, profiles = zmq_link.driver_link(args.host)

    if not experiment_io.config_numbers(args.exp_dir):
        print(f"Empty folder: seeding a {args.n}-point Latin hypercube design.")
        seed_folder(args)

    while True:
        configs = experiment_io.config_numbers(args.exp_dir)
        done = configs & experiment_io.run_numbers(args.exp_dir)
        pending = sorted(configs - done)
        if not pending:
            if len(done) >= args.iterations:
                break
            propose(args)
            continue

        # signal the controller to start on the head of the queue
        i = pending[0]
        payload = experiment_io.read_config(args.exp_dir, i) | dict(id=i)
        profiles.send_json(payload)

        # while the controller runs it, extend the queue against fantasized outcomes
        if len(done) >= 2 and len(configs) < args.iterations:
            propose(args)

        collected = zmq_link.collect_segment(
            samples, profiles, payload, args.samples, args.warmup
        )
        experiment_io.write_run(args.exp_dir, i, collected)
        print(f"segment {i}: {len(collected)} samples, loss = "
              f"{experiment_io.read_result(args.exp_dir, i):.6f}")

    print(f"Done: {args.iterations} segments in {args.exp_dir}.")


if __name__ == "__main__":
    main()
