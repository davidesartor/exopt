"""Mock drives script: control_loop.py on simulated legs fitted to the 2026-11-06 sessions, through a stand-in pyCandle."""

import argparse
import runpy
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from jaxtyping import Float

from exopt.control_loop import MAX_TORQUE, SAMPLING_STEP

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


def smoothstep(x: float) -> float:
    return x * x * (3 - 2 * x)


class Legs:
    """Hip trajectories fitted to real walking, pushed off course by the applied torque: I e'' + c e' + k e = torque.

    They start standing still on the zeroed pose, and walk after STAND_TIME."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.elapsed = 0.0

        # walking schedule
        self.walking, self.envelope, self.next_switch = False, 0.0, STAND_TIME

        # log cadence: treadmill speed level, slow drift, assistance felt
        self.level, self.smooth_level, self.drift, self.assistance = 0.0, 0.0, 0.0, 0.0
        self.next_speed_change = rng.exponential(SPEED_CHANGE_EVERY)

        # stride clock
        self.stride_angle, self.stride_period_scale = 0.0, 1.0
        self.stride_amplitudes: Float[np.ndarray, "2"] = rng.lognormal(0.0, STRIDE_AMPLITUDE_CV, 2)

        # reference trajectory, deviation from it, drives
        self.reference, self.reference_rate = np.zeros(2), np.zeros(2)
        self.deviation, self.deviation_rate = np.zeros(2), np.zeros(2)
        self.commanded = np.zeros(2)
        self.braking = False  # faulted drives ignore the command

    @property
    def position(self) -> Float[np.ndarray, "2"]:
        return self.reference + self.deviation

    @property
    def velocity(self) -> Float[np.ndarray, "2"]:
        return self.reference_rate + self.deviation_rate

    @property
    def applied(self) -> Float[np.ndarray, "2"]:
        """Torque on the legs: the command, or the faulted drives' passive brake."""
        return -BRAKE_DAMPING * self.velocity if self.braking else self.commanded

    @property
    def pace(self) -> float:
        """Log cadence relative to GAIT_PERIOD."""
        return self.smooth_level + self.drift + ADAPTATION * self.assistance

    def step(self, dt: float) -> None:
        """Advance the subject and the legs by dt [s]."""
        rng, applied = self.rng, self.applied
        self.elapsed += dt

        # walking schedule: stand, then walk, stopping now and then
        if self.elapsed >= self.next_switch:
            self.walking = not self.walking
            self.next_switch = self.elapsed + (
                rng.exponential(STOP_EVERY) if self.walking else rng.uniform(*STOP_DURATION))
        self.envelope = float(np.clip(self.envelope + (dt if self.walking else -dt) / WALK_RAMP, 0.0, 1.0))

        # log cadence: treadmill speed changes, slow drift, adaptation to the assistance felt
        if self.elapsed >= self.next_speed_change:
            self.level = float(np.clip(self.level + rng.normal(0.0, SPEED_CHANGE_SD), -MAX_PACE_SHIFT, MAX_PACE_SHIFT))
            self.next_speed_change += rng.exponential(SPEED_CHANGE_EVERY)
        self.smooth_level += dt / SPEED_CHANGE_TIME * (self.level - self.smooth_level)
        self.drift += -self.drift * dt / PACE_DRIFT_TIME + PACE_DRIFT_SD * np.sqrt(2 * dt / PACE_DRIFT_TIME) * rng.normal()
        self.assistance += dt / ADAPTATION_TIME * (float(np.mean(applied * self.velocity)) - self.assistance)
        pace = self.pace

        # stride clock, with a fresh period and amplitude scale every stride
        self.stride_angle += 2 * np.pi / GAIT_PERIOD * np.exp(pace) / self.stride_period_scale * dt
        if self.stride_angle >= 2 * np.pi:
            self.stride_angle -= 2 * np.pi
            self.stride_period_scale = rng.lognormal(0.0, STRIDE_PERIOD_CV)
            self.stride_amplitudes = np.array([self.stride_amplitudes[1], rng.lognormal(0.0, STRIDE_AMPLITUDE_CV)])
        stride_fraction = self.stride_angle / (2 * np.pi)
        amplitude = np.exp(AMPLITUDE_PER_CADENCE * pace) * (
            (1 - stride_fraction) * self.stride_amplitudes[0] + stride_fraction * self.stride_amplitudes[1])

        # reference hip angles, blended from the standing pose (encoders zeroed there)
        angles = self.stride_angle * np.arange(1, 5)
        shape = TRAJECTORY[:, 1::2] @ np.cos(angles) + TRAJECTORY[:, 2::2] @ np.sin(angles)
        reference = smoothstep(self.envelope) * (TRAJECTORY[:, 0] + amplitude * shape)
        self.reference, self.reference_rate = reference, (reference - self.reference) / dt

        # the applied torque pushes the legs off the reference
        restoring = HIP_DAMPING * self.deviation_rate + HIP_STIFFNESS * self.deviation
        self.deviation_rate = self.deviation_rate + dt * (applied - restoring) / LEG_INERTIA
        self.deviation = self.deviation + dt * self.deviation_rate

    def sense(self) -> Float[np.ndarray, "3 2"]:
        """Quantized encoder positions [rad], noisy velocities [rad/s] and measured torques [Nm]."""
        return np.stack([
            POSITION_QUANTUM * np.round(self.position / POSITION_QUANTUM),
            self.velocity + self.rng.normal(0.0, VELOCITY_NOISE, 2),
            self.applied + self.rng.normal(0.0, TORQUE_NOISE, 2),
        ])


class SimulatedCandle:
    """pyCandle.Candle stand-in whose two MD80 drives move simulated legs; readings refresh once per control step."""

    def __init__(self, seed: int | None, drive_fault: bool, drop_every: float | None):
        self.drive_fault, self.drop_every = drive_fault, drop_every
        self.legs = Legs(np.random.default_rng(seed))
        self.next_drop = None if drop_every is None else time.monotonic() + self.legs.rng.exponential(drop_every)
        self.reading = self.legs.sense()
        self.last_refresh = time.monotonic()
        self.stalled_until = 0.0
        self.md80s: list[SimpleNamespace] = []

    def refresh(self) -> None:
        """Move the legs for the time since the last refresh (stalls capped); reads freeze while the CAN link stalls."""
        now = time.monotonic()
        if now - self.last_refresh < SAMPLING_STEP / 2:
            return
        dt = min(now - self.last_refresh, 5 * SAMPLING_STEP) / SUBSTEPS
        self.last_refresh = now
        if self.next_drop is not None and now >= self.next_drop:
            self.next_drop = now + self.legs.rng.exponential(self.drop_every)
            self.drop("at random")
        for _ in range(SUBSTEPS):
            self.legs.step(dt)
        if now >= self.stalled_until:
            self.reading = self.legs.sense()

    def drop(self, why: str) -> None:
        """The CAN connection drops, as on 2026-11-06: reads stall, then both drives brake until re-enabled."""
        if self.legs.braking:
            return
        self.legs.braking = True
        self.stalled_until = time.monotonic() + self.legs.rng.uniform(*STALL_DURATION)
        print(f"mock dropped connection ({why}): reads stalled, drives braking passively until re-enabled")

    def get(self, quantity: int, leg: int) -> float:
        self.refresh()
        return float(self.reading[quantity, leg])

    def set_target_torque(self, leg: int, torque: float) -> None:
        if self.drive_fault and abs(torque) >= FAULT_TORQUE:
            self.drop("on a near-max command")
        self.legs.commanded = np.where(
            np.arange(2) == leg, np.clip(torque, -MAX_TORQUE, MAX_TORQUE), self.legs.commanded)

    def ping(self) -> list[int]:
        return [100, 101]

    def addMd80(self, id: int) -> None:
        leg = len(self.md80s)
        self.md80s.append(SimpleNamespace(
            getPosition=partial(self.get, 0, leg), getVelocity=partial(self.get, 1, leg),
            getTorque=partial(self.get, 2, leg), setTargetTorque=partial(self.set_target_torque, leg),
            setMaxTorque=lambda torque: None))

    def controlMd80Mode(self, id: int, mode: object) -> bool:
        return True

    def controlMd80Enable(self, id: int, on: bool) -> bool:
        if on:
            self.legs.braking = False
        return True

    def begin(self) -> None: ...

    def end(self) -> None: ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, help="seed of the simulated subject's variability")
    parser.add_argument("--drive-fault", action="store_true",
                        help="a command over 11 Nm drops the connection: reads stall, then the drives only brake until re-enabled")
    parser.add_argument("--drop-every", type=float, metavar="SECONDS",
                        help="the connection to the drives drops at random, this long apart on average")
    args = parser.parse_args()
    candle = SimulatedCandle(args.seed, args.drive_fault, args.drop_every)
    sys.modules["pyCandle"] = SimpleNamespace(Candle=lambda baud, printing: candle, CAN_BAUD_1M=None, RAW_TORQUE=None)
    runpy.run_path(str(Path(__file__).with_name("control_loop.py")), run_name="__main__")
