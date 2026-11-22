"""Metabolic tests: the exo held off, on power assist, or on an experiment's best profile, tagged in its stream."""

import uuid

import jax.numpy as jnp
from bofus.rkhs import RBFMixture

from exopt import experiment
from exopt.optimization_loop import best_trial
from exopt.control_loop import MAX_TORQUE

CONDITIONS = {"off": "exo off (zero torque)", "assist": "power assist (peak at the torque limit)",
              "best": "best profile (lowest posterior mean loss)"}


def steady_payload(exp_dir: str, condition: str) -> dict | None:
    """The profile held: None (zero torque) for off, power assist alone at the torque limit, or the trial of lowest
    posterior mean loss."""
    if condition == "off":
        return None
    settings = experiment.load_settings(exp_dir)
    common = dict(ramp_cycles=settings.ramp_cycles, stop_rule={})
    if condition == "assist":
        no_offset = RBFMixture(jnp.ones((1, 1)), jnp.zeros((1, 1)), jnp.zeros(1))
        return experiment.encode_profile(no_offset) | common | dict(
            id=f"0-metabolic-{uuid.uuid4().hex[:6]}", trial=None, power_assist=MAX_TORQUE)
    best = best_trial(experiment.load(exp_dir), settings)
    return experiment.encode_profile(best.profile) | common | dict(
        id=f"{best.index}-metabolic-{uuid.uuid4().hex[:6]}", trial=best.index, power_assist=best.power_assist)

