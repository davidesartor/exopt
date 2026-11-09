"""Mock controller script: controller.py on simulated legs fitted to the 2026-11-06 sessions, through a stand-in pyCandle."""

import argparse
import runpy
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
from jaxtyping import Float

from exopt import MAX_TORQUE
from exopt.controller import SAMPLING_STEP

# per-leg hip angle over a stride [rad]: mean, then (cos, sin) of harmonics 1-4
# fitted on 2026-11-06-atoms4 trials 4-19; leg 1's encoder is mirrored, so both fundamentals align
TRAJECTORY = np.array([
    [0.3575, 0.0397, 0.1445, -0.0118, -0.0071, 0.0088, -0.0036, -0.0008, -0.0003],
    [-0.2925, 0.0363, 0.1528, 0.0073, 0.0040, 0.0081, -0.0051, -0.0012, -0.0005],
])
GAIT_PERIOD = 1.165  # [s]
STRIDE_PERIOD_CV = 0.025  # stride-to-stride variability
STRIDE_AMPLITUDE_CV = 0.04
AMPLITUDE_PER_CADENCE = -0.5  # d log amplitude / d log cadence: on a treadmill, quicker steps are shorter
PACE_DRIFT_SD, PACE_DRIFT_TIME = 0.03, 300.0  # slow OU drift of log cadence, [s]
SPEED_CHANGE_EVERY, SPEED_CHANGE_SD, SPEED_CHANGE_TIME = 300.0, 0.15, 5.0  # [s], log cadence, [s]
MAX_PACE_SHIFT = 0.5  # bound on the log cadence level
STAND_TIME = 8.0  # [s] standing still at startup
STOP_EVERY = 600.0  # [s] mean walking time between stops
STOP_DURATION = (10.0, 40.0)  # [s]
WALK_RAMP = 2.0  # [s] to start or stop walking

LEG_INERTIA = 1.0  # [kg m^2] leg and exo about the hip
HIP_STIFFNESS = 500.0  # [Nm/rad] guess: 13 Nm moves the leg ~0.03 rad, a fifth of its swing amplitude
HIP_DAMPING = 15.0  # [Nm s/rad]
ADAPTATION, ADAPTATION_TIME = 0.01, 30.0  # guess: log cadence per W of assistance, lowpassed over [s]

POSITION_QUANTUM = 4.26e-5  # [rad] encoder resolution
VELOCITY_NOISE = 0.03  # [rad/s] sd, measured standing still
TORQUE_NOISE = 0.05  # [Nm] sd of the measured torque
FAULT_TORQUE = 11.0  # [Nm] command that trips the drive fault (when enabled)
STALL_DURATION = (0.3, 0.9)  # [s] frozen reads as the drives fault
BRAKE_DAMPING = 3.4  # [Nm s/rad] faulted drives: measured torque = -BRAKE_DAMPING * velocity
SUBSTEPS = 10  # integration substeps per control step


class Legs(NamedTuple):
    """Hip trajectories fitted to real walking, pushed off course by the applied torque: I e'' + c e' + k e = torque."""

    elapsed: float

    # gait: walking schedule, cadence, stride clock
    walking: bool
    envelope: float
    next_switch: float
    level: float
    smooth_level: float
    drift: float
    assistance: float
    next_speed_change: float
    stride_angle: float
    stride_period_scale: float
    stride_amplitudes: Float[np.ndarray, "2"]

    # legs: reference, deviation from it, torques, drive state
    reference: Float[np.ndarray, "2"]
    reference_rate: Float[np.ndarray, "2"]
    deviation: Float[np.ndarray, "2"]
    deviation_rate: Float[np.ndarray, "2"]
    commanded: Float[np.ndarray, "2"]
    applied: Float[np.ndarray, "2"]
    braking: bool


def smoothstep(x: float) -> float:
    return x * x * (3 - 2 * x)


def standing_legs(rng: np.random.Generator) -> Legs:
    """Standing still on the zeroed pose, to start walking after STAND_TIME."""
    zeros = np.zeros(2)
    return Legs(
        elapsed=0.0, walking=False, envelope=0.0, next_switch=STAND_TIME,
        level=0.0, smooth_level=0.0, drift=0.0, assistance=0.0,
        next_speed_change=rng.exponential(SPEED_CHANGE_EVERY),
        stride_angle=0.0, stride_period_scale=1.0, stride_amplitudes=rng.lognormal(0.0, STRIDE_AMPLITUDE_CV, 2),
        reference=zeros, reference_rate=zeros, deviation=zeros, deviation_rate=zeros,
        commanded=zeros, applied=zeros, braking=False,
    )


def step_legs(legs: Legs, dt: float, rng: np.random.Generator) -> Legs:
    """Advance the subject and the legs by dt [s]."""
    elapsed = legs.elapsed + dt

    # walking schedule: stand, then walk, stopping now and then
    walking, next_switch = legs.walking, legs.next_switch
    if elapsed >= next_switch:
        walking = not walking
        next_switch = elapsed + (rng.exponential(STOP_EVERY) if walking else rng.uniform(*STOP_DURATION))
    envelope = float(np.clip(legs.envelope + (dt if walking else -dt) / WALK_RAMP, 0.0, 1.0))

    # log cadence: treadmill speed changes, slow drift, adaptation to the assistance felt
    level, next_speed_change = legs.level, legs.next_speed_change
    if elapsed >= next_speed_change:
        level = float(np.clip(level + rng.normal(0.0, SPEED_CHANGE_SD), -MAX_PACE_SHIFT, MAX_PACE_SHIFT))
        next_speed_change += rng.exponential(SPEED_CHANGE_EVERY)
    smooth_level = legs.smooth_level + dt / SPEED_CHANGE_TIME * (level - legs.smooth_level)
    drift = legs.drift - legs.drift * dt / PACE_DRIFT_TIME + PACE_DRIFT_SD * np.sqrt(
        2 * dt / PACE_DRIFT_TIME
    ) * rng.normal()
    velocity = legs.reference_rate + legs.deviation_rate
    power = float(np.mean(legs.applied * velocity))
    assistance = legs.assistance + dt / ADAPTATION_TIME * (power - legs.assistance)
    pace = smooth_level + drift + ADAPTATION * assistance

    # stride clock, with a fresh period and amplitude scale every stride
    omega = 2 * np.pi / GAIT_PERIOD * np.exp(pace) / legs.stride_period_scale
    stride_angle = legs.stride_angle + omega * dt
    stride_period_scale, stride_amplitudes = legs.stride_period_scale, legs.stride_amplitudes
    if stride_angle >= 2 * np.pi:
        stride_angle -= 2 * np.pi
        stride_period_scale = rng.lognormal(0.0, STRIDE_PERIOD_CV)
        stride_amplitudes = np.array([stride_amplitudes[1], rng.lognormal(0.0, STRIDE_AMPLITUDE_CV)])
    stride_fraction = stride_angle / (2 * np.pi)
    amplitude = np.exp(AMPLITUDE_PER_CADENCE * pace) * (
        (1 - stride_fraction) * stride_amplitudes[0] + stride_fraction * stride_amplitudes[1]
    )

    # reference hip angles, blended from the standing pose (encoders zeroed there)
    angles = stride_angle * np.arange(1, 5)
    shape = TRAJECTORY[:, 1::2] @ np.cos(angles) + TRAJECTORY[:, 2::2] @ np.sin(angles)
    reference = smoothstep(envelope) * (TRAJECTORY[:, 0] + amplitude * shape)

    # the command, or the faulted drives' passive brake, pushes the legs off the reference
    applied = -BRAKE_DAMPING * velocity if legs.braking else legs.commanded
    restoring = HIP_DAMPING * legs.deviation_rate + HIP_STIFFNESS * legs.deviation
    deviation_rate = legs.deviation_rate + dt * (applied - restoring) / LEG_INERTIA

    return legs._replace(
        elapsed=elapsed, walking=walking, envelope=envelope, next_switch=next_switch,
        level=level, smooth_level=smooth_level, drift=drift, assistance=assistance,
        next_speed_change=next_speed_change, stride_angle=stride_angle,
        stride_period_scale=stride_period_scale, stride_amplitudes=stride_amplitudes,
        reference=reference, reference_rate=(reference - legs.reference) / dt,
        deviation=legs.deviation + dt * deviation_rate, deviation_rate=deviation_rate, applied=applied,
    )


def sense(legs: Legs, rng: np.random.Generator) -> Float[np.ndarray, "3 2"]:
    """Quantized encoder positions [rad], noisy velocities [rad/s] and measured torques [Nm]."""
    return np.stack([
        POSITION_QUANTUM * np.round((legs.reference + legs.deviation) / POSITION_QUANTUM),
        legs.reference_rate + legs.deviation_rate + rng.normal(0.0, VELOCITY_NOISE, 2),
        legs.applied + rng.normal(0.0, TORQUE_NOISE, 2),
    ])


def simulated_pycandle(seed: int | None, drive_fault: bool) -> SimpleNamespace:
    """pyCandle stand-in whose two MD80 drives move simulated legs; readings refresh once per control step."""
    rng = np.random.default_rng(seed)
    legs = standing_legs(rng)
    reading = sense(legs, rng)
    last_refresh = time.monotonic()
    stalled_until = 0.0

    def refresh() -> None:
        # the legs move for the time since the last refresh (stalls capped); reads freeze while the CAN link stalls
        nonlocal legs, reading, last_refresh
        now = time.monotonic()
        if now - last_refresh < SAMPLING_STEP / 2:
            return
        dt = min(now - last_refresh, 5 * SAMPLING_STEP) / SUBSTEPS
        last_refresh = now
        for _ in range(SUBSTEPS):
            legs = step_legs(legs, dt, rng)
        if now >= stalled_until:
            reading = sense(legs, rng)

    def get(quantity: int, leg: int) -> float:
        refresh()
        return float(reading[quantity, leg])

    def set_target_torque(leg: int, torque: float) -> None:
        # as on 2026-11-06: a near-max command stalls the reads, then both drives brake for good
        nonlocal legs, stalled_until
        if drive_fault and not legs.braking and abs(torque) >= FAULT_TORQUE:
            legs = legs._replace(braking=True)
            stalled_until = time.monotonic() + rng.uniform(*STALL_DURATION)
            print("drive fault: reads stalled, drives braking passively from now on")
        commanded = np.where(np.arange(2) == leg, np.clip(torque, -MAX_TORQUE, MAX_TORQUE), legs.commanded)
        legs = legs._replace(commanded=commanded)

    def md80(leg: int) -> SimpleNamespace:
        return SimpleNamespace(
            getPosition=partial(get, 0, leg), getVelocity=partial(get, 1, leg), getTorque=partial(get, 2, leg),
            setTargetTorque=partial(set_target_torque, leg), setMaxTorque=lambda torque: None,
        )

    md80s = []
    candle = SimpleNamespace(
        md80s=md80s, ping=lambda: [100, 101], addMd80=lambda id: md80s.append(md80(len(md80s))),
        controlMd80Mode=lambda id, mode: True, controlMd80Enable=lambda id, enable: True,
        begin=lambda: None, end=lambda: None,
    )
    return SimpleNamespace(Candle=lambda baud, printing: candle, CAN_BAUD_1M=None, RAW_TORQUE=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, help="seed of the simulated subject's variability")
    parser.add_argument("--drive-fault", action="store_true",
                        help="the first command over 11 Nm stalls the reads, then the drives only brake")
    args = parser.parse_args()
    sys.modules["pyCandle"] = simulated_pycandle(args.seed, args.drive_fault)
    runpy.run_path(str(Path(__file__).with_name("controller.py")), run_name="__main__")
