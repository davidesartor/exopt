"""bo-loop: propose a torque profile, run it on the controller, record it, repeat."""

from functools import partial
from jaxtyping import Array, Float, Key

import jax
import jax.numpy as jnp
import jax.random as jr

import argparse
import os
import signal
import threading
import time
import uuid

import numpy as np
import zmq

from einops import rearrange
from bofus import acquisition
from bofus.gp import GaussianProcess
from bofus.rkhs import X_MAX, RBFMixture
from bofus.utils import latin_hypercube_sample, rescale

from exopt import experiment, streams
from exopt.control_loop import MAX_TORQUE, PERIODIC
from exopt.review import write_review

MAX_CAPACITY = 1024  # largest history capacity compiled ahead


def variance_range(settings: experiment.Settings) -> tuple[Float[Array, ""], Float[Array, ""]]:
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


def capacity(trials_count: int) -> int:
    """History length a proposal is padded to: a power of 2, at least 8."""
    return max(8, 1 << (trials_count - 1).bit_length())


def stack(
    profiles: list[RBFMixture], losses: list[float], ses: list[float]
) -> tuple[RBFMixture, Float[Array, "n"], Float[Array, "n"]]:
    """One batched history, nan-padded to a power-of-2 length so proposals recompile only when it doubles."""
    m = max(p.a.shape[-1] for p in profiles)
    n = capacity(len(profiles))
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


@jax.jit
def posterior_means(key: Key, history: RBFMixture, ys: Float[Array, "n"], y_stds: Float[Array, "n"]) -> Float[Array, "n"]:
    """Posterior mean loss at each profile of the history, infinite at the padding."""
    f = jax.tree.map(lambda z: rearrange(z, "n ... -> n 1 ..."), history)
    surrogate = GaussianProcess.fit(key, f, ys, y_std=y_stds, periodic=PERIODIC)
    mean, _ = surrogate.predict_marginal(surrogate.f)
    return jnp.where(jnp.isfinite(ys), jnp.ravel(mean), jnp.inf)


def best_trial(history: list[experiment.Trial], settings: experiment.Settings) -> experiment.Trial:
    """The trial whose profile has the lowest posterior mean loss."""
    stacked = stack([t.profile for t in history], [t.loss for t in history], [t.se for t in history])
    return history[int(posterior_means(jr.key(settings.seed), *stacked).argmin())]


def compile_ahead(history: list[experiment.Trial], settings: experiment.Settings) -> None:
    """Compile the proposal for each coming history capacity, so none stalls the session; jit reuses them."""
    profiles = [t.profile for t in history] or [jax.tree.map(lambda z: z[0], design(
        jr.key(0), settings.atoms, 1, variance_range(settings)))]
    size = capacity(max(len(history) + 1, settings.n))
    while size <= MAX_CAPACITY:
        propose_next.lower(
            jr.key(0), *stack((profiles * size)[:size], [0.0] * size, [0.0] * size), variance_range(settings),
            atoms=settings.atoms, raw_samples=settings.raw_samples, max_restarts=settings.max_restarts,
        ).compile()
        size *= 2


def next_profile(key: Key, history: list[experiment.Trial], settings: experiment.Settings) -> tuple[RBFMixture, dict | None]:
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
    for name, default in experiment.DEFAULT._asdict().items():
        label, hint = experiment.SETTINGS_HELP[name]
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(default), default=default,
                            help=f"{label}{f': {hint}' if hint else ''} (default {default})")
    args = parser.parse_args()
    settings = experiment.Settings(**{name: getattr(args, name) for name in experiment.Settings._fields})
    # an SE needs 2 gaits
    settings = settings._replace(min_gaits=max(settings.min_gaits, 2))
    settings = settings._replace(max_gaits=max(settings.max_gaits, settings.min_gaits))
    if not abs(settings.power_assist) < MAX_TORQUE:
        parser.error(f"--power-assist must be below the {MAX_TORQUE:g} Nm torque limit")
    # every trial of the folder is scored under the settings it last ran with
    experiment.save_settings(args.exp_dir, settings)

    # SIGINT stops the session cleanly even when launched with it ignored (a background job)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    if os.name == "nt":  # the dashboard's stop on Windows is a ctrl-break
        signal.signal(signal.SIGBREAK, signal.default_int_handler)

    samples, profiles = streams.driver_link(args.host)
    try:
        run_session(samples, profiles, args.exp_dir, settings)
    except ConnectionError as error:
        print(f"Stopped: {error}")
    finally:
        streams.release(profiles)
        print(f"Review in {write_review(args.exp_dir)}")


def collect_segment(samples: zmq.Socket, profiles: zmq.Socket, payload: dict, settings: experiment.Settings) -> tuple[list[dict], str]:
    """Publish the profile and gather samples until payload["stop_rule"] stops it (experiment.stop_reason), and why.

    The warm-up counts from the end of the controller's ramp up; valid gaits (experiment.gait_slices) count after it. The segment
    returned ends with a gait. Raises ConnectionError after streams.STALL seconds without samples (the controller streams from
    startup).
    """
    collected, phase = [], None
    rule = payload["stop_rule"]
    published_at, heard_at = None, time.monotonic()
    while True:
        # republish every second until the profile is fully ramped up (slow-joiner losses)
        if not collected and (published_at is None or time.monotonic() - published_at > 1.0):
            profiles.send_json(payload)
            published_at = time.monotonic()
        if not samples.poll(100):
            if time.monotonic() - heard_at > streams.STALL:
                raise ConnectionError(f"no samples from the controller for {streams.STALL:g} s")
            continue

        # keep only samples of the requested profile, once ramped up to full gain
        sample = samples.recv_json()
        heard_at = time.monotonic()
        if sample["profile_id"] != payload["id"] or sample["gain"] < 1.0:
            continue
        collected.append(sample)
        if len(collected) <= settings.warmup:
            continue

        # a gait ends where leg 0's phase wraps: check the rule on the segment up to it
        wrapped = None not in (phase, sample["phase_0"]) and sample["phase_0"] - phase < -np.pi
        phase = sample["phase_0"]
        if wrapped:
            segment = collected[settings.warmup:-1]
            if reason := experiment.stop_reason(*experiment.gait_estimate(experiment.columns(segment), settings), rule, settings):
                return segment, reason


def run_session(samples, profiles, exp_dir: str, settings: experiment.Settings) -> None:
    """Propose, ramp up, collect, release, record, until interrupted or the controller is gone."""
    # compiled while the first trials run (the initial design needs none)
    threading.Thread(target=compile_ahead, args=(experiment.load(exp_dir), settings), daemon=True).start()
    while True:
        history = experiment.load(exp_dir)
        profile, best = next_profile(jr.key(settings.seed), history, settings)

        # a fresh id per publish, so the controller never mistakes it for a stale profile
        index = max((t.index for t in history), default=0) + 1
        payload = experiment.encode_profile(profile) | dict(id=f"{index}-{uuid.uuid4().hex[:6]}",
                                                        ramp_cycles=settings.ramp_cycles, power_assist=settings.power_assist,
                                                        stop_rule=best or {})
        experiment.save_current(exp_dir, payload)
        collected, reason = collect_segment(samples, profiles, payload, settings)
        # ramp down to zero while the next proposal is computed
        streams.release(profiles)
        trial = experiment.save(exp_dir, index, profile, collected, settings)
        best = min([t.loss for t in history] + [trial.loss])
        print(f"trial {index}: {len(experiment.gait_slices(trial.samples, settings))} valid gaits ({reason}), loss {trial.loss:.4f} ± {trial.se:.4f} (best {best:.4f})")


if __name__ == "__main__":
    main()
