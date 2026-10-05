"""exo-controller: estimate the gait phase, apply the published torque profile, stream samples."""

import argparse
import sys
import threading
import time

import jax.numpy as jnp
import numpy as np

from exopt import MAX_TORQUE, link
from exopt.profiles import Profile

SAMPLING_STEP = 0.01  # [s] control loop period
MIN_CALIBRATION_TIME = 10.0  # [s] minimum time to calibrate the gait frequency
CALIBRATION_TOL = 0.01  # relative drift of omega per second at convergence
LOWPASS_SMOOTHING = 0.1  # EMA weight before the phase estimation
GAIN_RAMP_TIME = 5.0  # [s] blend from the previous profile to the new one after a swap

# simulated subject
TRUE_GAIT_PERIOD = 1.0  # [s] one gait cycle (2 steps, 1 per leg)
LEG_INERTIA = 1.0  # [kg m^2] leg about the hip
HIP_STIFFNESS = 150.0  # [Nm/rad] subject's impedance around the reference gait
HIP_DAMPING = 5.0  # [Nm s/rad]
SUBSTEPS = 10  # integration substeps per control step


class HardwareLegs:
    """Two MD80 drives over CAN in raw torque mode, zeroed with the subject standing still."""

    def __init__(self):
        import pyCandle

        input("Subject standing still: press enter to zero the position")
        self.candle = candle = pyCandle.Candle(pyCandle.CAN_BAUD_1M, True)
        self.ids = candle.ping()[:2]
        if len(self.ids) < 2:
            sys.exit(f"expected 2 motors, found {len(self.ids)}")
        for id in self.ids:
            candle.addMd80(id)
        for id, md in zip(self.ids, candle.md80s):
            if not candle.controlMd80SetEncoderZero(id):
                sys.exit(f"failed to zero encoder on drive {id}")
            if not candle.controlMd80Mode(id, pyCandle.RAW_TORQUE):
                sys.exit(f"failed to set mode on drive {id}")
            if not candle.controlMd80Enable(id, True):
                sys.exit(f"failed to enable drive {id}")
            md.setMaxTorque(MAX_TORQUE)
        candle.begin()

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Positions [rad], velocities [rad/s] and measured torques [Nm] of both legs."""
        mds = self.candle.md80s
        return (
            np.array([md.getPosition() for md in mds]),
            np.array([md.getVelocity() for md in mds]),
            np.array([md.getTorque() for md in mds]),
        )

    def command(self, torque: np.ndarray) -> None:
        for md, tau in zip(self.candle.md80s, torque):
            md.setTargetTorque(float(tau))

    def close(self) -> None:
        """Leave the motors unpowered."""
        self.command(np.zeros(2))
        self.candle.end()
        for id in self.ids:
            self.candle.controlMd80Enable(id, False)


class SimulatedLegs:
    """Legs following a reference gait through a hip impedance: I e'' + c e' + k e = torque."""

    def __init__(self):
        self.start = self.last_read = time.monotonic()
        self.deviation = self.deviation_rate = self.torque = np.zeros(2)

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        now = time.monotonic()

        # the legs respond to the torque applied since the last read (stalls capped)
        dt = min(now - self.last_read, 5 * SAMPLING_STEP) / SUBSTEPS
        self.last_read = now
        for _ in range(SUBSTEPS):
            restoring = HIP_DAMPING * self.deviation_rate + HIP_STIFFNESS * self.deviation
            self.deviation_rate = self.deviation_rate + dt * (self.torque - restoring) / LEG_INERTIA
            self.deviation = self.deviation + dt * self.deviation_rate

        # hip-like reference: dominant fundamental plus an asymmetric second harmonic
        omega = 2 * np.pi / TRUE_GAIT_PERIOD
        phase = omega * (now - self.start) + np.array([0.0, np.pi])
        position = 0.5 * np.cos(phase) + 0.1 * np.sin(2 * phase + 0.5)
        velocity = omega * (-0.5 * np.sin(phase) + 0.2 * np.cos(2 * phase + 0.5))
        return position + self.deviation, velocity + self.deviation_rate, self.torque

    def command(self, torque: np.ndarray) -> None:
        self.torque = torque

    def close(self) -> None:
        pass


class PhaseEstimator:
    """Gait phase arctan2(-v/omega, x), with omega^2 = <v^2>/<x^2> calibrated until stable."""

    def __init__(self):
        self.calibrated = False
        self.n, self.offset = 0, np.zeros(2)
        self.sum_sq_position = self.sum_sq_velocity = 0.0
        self.omega = self.checked_omega = self.started_at = self.checked_at = None

    def calibrate(self, now: float, position: np.ndarray, velocity: np.ndarray) -> None:
        """Accumulate the stride-frequency statistics; calibrated once omega drifts < tol per second."""
        if self.started_at is None:
            self.started_at = self.checked_at = now

        # center position on its running mean (the encoder zero is unreliable)
        self.n += 1
        self.offset = self.offset + (position - self.offset) / self.n
        self.sum_sq_position += float(np.sum((position - self.offset) ** 2))
        self.sum_sq_velocity += float(np.sum(velocity**2))
        if self.sum_sq_position > 0.0:
            self.omega = (self.sum_sq_velocity / self.sum_sq_position) ** 0.5

        # once per second, check the drift of the estimate
        if now - self.checked_at < 1.0:
            return
        settled = self.omega is not None and self.checked_omega is not None
        drift = abs(self.omega - self.checked_omega) / self.omega if settled else np.inf
        self.checked_omega, self.checked_at = self.omega, now
        if now - self.started_at >= MIN_CALIBRATION_TIME and drift < CALIBRATION_TOL:
            self.calibrated = True
            # offset EMA with an effective sample size of one gait cycle
            self.offset_smoothing = SAMPLING_STEP * self.omega / (2 * np.pi)
            self.smooth_position, self.smooth_velocity = position - self.offset, velocity

    def __call__(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Phase [rad] of both legs from the lowpassed, centered state."""
        self.offset = self.offset + self.offset_smoothing * (position - self.offset)
        self.smooth_position = self.smooth_position + LOWPASS_SMOOTHING * (
            position - self.offset - self.smooth_position
        )
        self.smooth_velocity = self.smooth_velocity + LOWPASS_SMOOTHING * (
            velocity - self.smooth_velocity
        )
        return np.arctan2(-self.smooth_velocity / self.omega, self.smooth_position)


def per_leg(**columns: np.ndarray | None) -> dict:
    """Flatten two-leg arrays into name_0 / name_1 sample fields."""
    return {
        f"{name}_{leg}": None if values is None else float(values[leg])
        for name, values in columns.items()
        for leg in (0, 1)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="simulated legs instead of the drives")
    parser.add_argument("--duration", type=float, default=30.0, help="experiment time [min]")
    args = parser.parse_args()

    samples, profiles = link.controller_link()
    legs = SimulatedLegs() if args.mock else HardwareLegs()

    # stream from now on; calibration starts once the subject walks
    walking = threading.Event()
    if args.mock:
        walking.set()
    else:
        threading.Thread(target=lambda: (input("Press enter to start walking\n"), walking.set()), daemon=True).start()

    estimator = PhaseEstimator()
    zero = Profile(jnp.zeros(1), jnp.zeros(1))
    payload, profile, previous = dict(id=None), zero, zero
    start = swap_at = next_tick = time.monotonic()
    stage = "standing"
    try:
        while (now := time.monotonic()) - start < 60.0 * args.duration:
            position, velocity, measured_torque = legs.read()
            if walking.is_set() and not estimator.calibrated:
                stage = "calibrating"
                estimator.calibrate(now, position, velocity)
                if estimator.calibrated:
                    stage = "running"
                    print(f"calibrated: gait period {2 * np.pi / estimator.omega:.3f} s")

            # swap in a new profile once calibrated (republished ids are ignored)
            update = link.latest_message(profiles) if estimator.calibrated else None
            if update is not None and update["id"] != payload["id"]:
                payload, previous, profile, swap_at = update, profile, Profile.from_dict(update), now
                print(f"profile {payload['id']}")

            # blend from the previous profile to the new one; zero torque until calibrated
            blend = min((now - swap_at) / GAIN_RAMP_TIME, 1.0)
            phase, torque = None, np.zeros(2)
            if estimator.calibrated:
                phase = estimator(position, velocity)
                fraction = blend * profile(phase) + (1 - blend) * previous(phase)
                torque = MAX_TORQUE * np.clip(np.asarray(fraction), -1.0, 1.0)
            legs.command(torque)

            samples.send_json(
                dict(time=now - start, stage=stage, profile_id=payload["id"], blend=blend)
                | per_leg(
                    phase=phase,
                    position=position,
                    velocity=velocity,
                    torque=torque,
                    measured_torque=measured_torque,
                )
            )
            next_tick = max(next_tick + SAMPLING_STEP, time.monotonic())
            time.sleep(max(next_tick - time.monotonic(), 0.0))
    finally:
        legs.close()


if __name__ == "__main__":
    main()
