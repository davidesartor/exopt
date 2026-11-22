"""Control loop script: estimate the gait phase, apply the published torque profile to the drives, stream samples."""

import signal
import time

import jax
import jax.numpy as jnp
import numpy as np
from bofus.rkhs import X_MAX, RBFMixture
from einops import rearrange
from jaxtyping import Float

from exopt import streams
from exopt.gait_estimation import PhaseEstimator
from exopt.drives_md80 import MAX_TORQUE, Drives

PERIODIC = (True,)  # a profile's one input, the gait phase, wraps
SAMPLING_STEP = 0.01  # [s] control loop period
STILL_SPEED = 0.15  # [rad/s] both legs slower than this count as standing still (velocity noise sd 0.03)
STILL_TIME = 1.0  # [s] standing still before the logged positions are re-zeroed on the standing pose
IDLE_TIMEOUT = 300.0  # [s] with nothing connected before the controller exits
GAIN_RAMP_CYCLES = 2.0  # [gait cycles] ramp of the profile gain between zero and full scale, unless requested
PROFILE_GRID = np.linspace(0.0, 2 * np.pi, 1025)  # [rad] phases a profile is tabulated at, interpolated per tick


def unit_phase(phase: Float[np.ndarray, "..."]) -> Float[np.ndarray, "..."]:
    """Gait phase [rad] on [0, 2pi) to the profile domain [-X_MAX, X_MAX)."""
    return X_MAX * (np.asarray(phase) / np.pi - 1.0)


def smooth_ramp(progress: float) -> float:
    """Gain along a ramp: smoothstep of the linear progress on [0, 1], so torque starts and ends with zero slope."""
    return progress * progress * (3.0 - 2.0 * progress)


def power_assist_torque(peak: float, phase: Float[np.ndarray, "p"]) -> Float[np.ndarray, "p"]:
    """Torque [Nm] in phase with the velocity (position ~ cos(phase)), peaking at `peak` where the position is 0."""
    return -peak * np.sin(phase)


def encode_profile(profile: RBFMixture) -> dict:
    return {k: v.tolist() for k, v in profile._asdict().items()}


def decode_profile(payload: dict) -> RBFMixture:
    return RBFMixture(*(jnp.asarray(payload[k], dtype=float) for k in RBFMixture._fields))


def profile_torque(
    profile: RBFMixture, phase: Float[np.ndarray, "p"], power_assist: float = 0.0
) -> Float[np.ndarray, "p"]:
    """Torque [Nm] of a profile at gait phases [rad]: MAX_TORQUE * tanh(base + mixture).

    The base is the power assist sinusoid of peak `power_assist` before the tanh, so a zero mixture applies it exactly."""
    values = jax.vmap(lambda x: profile(x, PERIODIC))(rearrange(unit_phase(phase), "p -> p 1"))
    # a peak at MAX_TORQUE puts the base at ±inf, where the tanh saturates exactly
    with np.errstate(divide="ignore"):
        base = np.arctanh(power_assist_torque(power_assist, phase) / MAX_TORQUE)
    return MAX_TORQUE * np.tanh(base + np.asarray(values))


class ProfilePlayer:
    """The requested profile behind a gain ramping between zero and full scale, so each trial starts alike."""

    def __init__(self):
        self.requested: dict = dict(id=None)  # a profile, or a release (id None)
        self.applied_id: str | None = None
        self.gain = 0.0
        self.ramp_cycles = GAIN_RAMP_CYCLES
        self.table = np.zeros_like(PROFILE_GRID)

    @property
    def wanted(self) -> bool:
        """The applied profile is the one still requested."""
        return self.requested["id"] is not None and self.requested["id"] == self.applied_id

    def request(self, message: dict) -> None:
        """Take a request; republished ids are ignored."""
        if message["id"] == self.requested["id"]:
            return
        self.requested = message
        self.ramp_cycles = message.get("ramp_cycles", self.ramp_cycles)
        print(f"profile {message['id']}" if message["id"] else "release")

    def advance(self, cycles: float, engaged: bool) -> None:
        """Move the gain by `cycles` of gait: up while engaged on the wanted profile, down otherwise."""
        # profiles swap only at zero gain
        if self.gain == 0.0 and self.requested["id"] not in (None, self.applied_id):
            self.applied_id = self.requested["id"]
            self.table = profile_torque(decode_profile(self.requested), PROFILE_GRID,
                                        self.requested.get("power_assist", 0.0))
        target, step = float(self.wanted and engaged), cycles / self.ramp_cycles
        self.gain = min(self.gain + step, target) if self.gain < target else max(self.gain - step, target)

    def torque(self, phase: Float[np.ndarray, "2"]) -> Float[np.ndarray, "2"]:
        return smooth_ramp(self.gain) * np.interp(phase, PROFILE_GRID, self.table, period=2 * np.pi)


def main() -> None:
    """Stream samples and apply the requested profiles until ctrl-c, or IDLE_TIMEOUT with nothing connected."""
    # a dropped ssh session ends the run like ctrl-c, so the drives are released
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.default_int_handler)

    with Drives() as drives:
        samples, profiles = streams.controller_link()
        peers = streams.Peers(samples, profiles)
        estimator, player = PhaseEstimator(SAMPLING_STEP), ProfilePlayer()
        start = last_tick = next_tick = idle_since = still_since = time.monotonic()
        stage, standing_pose = None, None

        while True:
            now = time.monotonic()
            if peers():
                idle_since = now
            elif now - idle_since > IDLE_TIMEOUT:
                print(f"nothing connected for {IDLE_TIMEOUT / 60:g} min: exiting")
                break
            raw_position, velocity, measured_torque = drives.read(now - start)

            # logged positions are zeroed on the standing pose, re-taken whenever the subject stands still
            if np.any(np.abs(velocity) > STILL_SPEED):
                still_since = now
            if standing_pose is None or now - still_since >= STILL_TIME:
                standing_pose = raw_position

            # calibration starts by itself once strides show up
            estimator.update(now, raw_position, velocity)
            phase = estimator.phase
            previous_stage = stage
            stage = "standing" if not estimator.walking else "running" if phase is not None else "calibrating"
            if stage != previous_stage:
                print(f"running: gait period {2 * np.pi / estimator.omega:.3f} s" if stage == "running" else stage)

            # zero torque until calibrated, and ramped down while the subject stands
            torque = np.zeros(2)
            if phase is not None:
                if (message := streams.latest_message(profiles)) is not None:
                    player.request(message)
                player.advance((now - last_tick) * estimator.omega / (2 * np.pi), engaged=stage == "running")
                torque = player.torque(phase)
            last_tick = now
            drives.apply(torque)

            columns = dict(phase=phase, position=raw_position - standing_pose, velocity=velocity, torque=torque,
                           measured_torque=measured_torque)
            samples.send_json(
                dict(time=now - start, stage=stage, profile_id=player.applied_id, gain=player.gain)
                | {f"{name}_{leg}": None if values is None else float(values[leg])
                   for name, values in columns.items() for leg in (0, 1)}
            )
            next_tick = max(next_tick + SAMPLING_STEP, time.monotonic())
            time.sleep(max(next_tick - time.monotonic(), 0.0))


if __name__ == "__main__":
    main()
