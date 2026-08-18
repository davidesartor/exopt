"""Hardware controller: drive the exo motors over pyCandle and stream samples over ZMQ."""

import sys
import time
import pyCandle
import jax.numpy as jnp

from jaxtyping import Array, Float
from tqdm import tqdm
from exopt import zmq_link

# control parameters
EXPERIMENT_TIME = 60.0  # [s] maximum time to run the experiment
CALIBRATION_TIME = 5.0  # [s] time to calibrate the gait period
SAMPLING_RATE = 100.0  # [Hz] control loop frequency
SAMPLING_STEP = 1.0 / SAMPLING_RATE  # [s] control loop period
TORQUE_SCALE = 5.0  # [Nm] scale factor for the torque
LOWPASS_SMOOTHING = 0.5  # EMA weight before the phase-estimation
MAX_TORQUE = 13.0  # [Nm] hardware torque limit per motor
GAIN_RAMP_TIME = 5.0  # [s] ramp assistance from 0 to full after a profile swap


def setup_motors() -> pyCandle.Candle:
    """Connect over CAN, zero encoders, and enable both motors in raw torque mode."""
    candle = pyCandle.Candle(pyCandle.CAN_BAUD_1M, True)
    ids = candle.ping()
    if len(ids) < 2:
        sys.exit(f"expected 2 motors, found {len(ids)}")
    for id in ids[:2]:
        candle.addMd80(id)
        candle.controlMd80SetEncoderZero(id)
        candle.controlMd80Mode(id, pyCandle.RAW_TORQUE)
        candle.controlMd80Enable(id, True)
    for md in candle.md80s:
        md.setMaxTorque(MAX_TORQUE)
    return candle


def read_legs_state(
    candle: pyCandle.Candle,
) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
    """Leg positions [rad] and angular velocities [rad/s] from the encoders."""
    position = jnp.array([md.getPosition() for md in candle.md80s])
    velocity = jnp.array([md.getVelocity() for md in candle.md80s])
    return position, velocity


def apply_torques(candle: pyCandle.Candle, torque: Float[Array, "2"]) -> None:
    """Send the target torque [Nm] to each motor."""
    for md, tau in zip(candle.md80s, torque):
        md.setTargetTorque(float(tau))


if __name__ == "__main__":
    # set up ZMQ sockets for streaming samples and receiving profiles
    samples, profiles = zmq_link.controller_link()
    candle = setup_motors()
    candle.begin()
    input("Press enter to start")
    start = time.monotonic()

    try:
        # estimate the stride frequency (omega^2 = <velocity^2>/<position^2>)
        with tqdm(
            total=CALIBRATION_TIME, desc="calibrating gait period", unit="s"
        ) as pbar:
            mean_sq_position, mean_sq_velocity = 0.0, 0.0
            gait_omega = None
            while (now := time.monotonic()) - start < CALIBRATION_TIME:
                # accumulate mean squared position and velocity over both legs
                position, velocity = read_legs_state(candle)
                mean_sq_position += float(jnp.sum(position**2))
                mean_sq_velocity += float(jnp.sum(velocity**2))
                gait_omega = jnp.sqrt(mean_sq_velocity / mean_sq_position)

                # control loop step
                pbar.n = now - start
                pbar.set_postfix(ω=f"{float(gait_omega):.4f} rad/s")
                time.sleep(SAMPLING_STEP)

        # report the estimated gait period and angular velocity
        assert gait_omega is not None, "failed to calibrate gait period"
        gait_period = 2 * jnp.pi / gait_omega
        print(f"ω = {float(gait_omega):.2f} rad/s")
        print(f"τ = {float(gait_period):.4f} s")

        # block until the first profile arrives
        print(f"Waiting for a profile to stream samples...")
        while (payload := zmq_link.latest_torque_profile(profiles)) is None:
            time.sleep(SAMPLING_STEP)
        torque_profile = zmq_link.config_torque_profile(payload)
        smooth_position, smooth_velocity = read_legs_state(candle)

        # run the experiment, streaming samples and updating the profile when a new one arrives
        with tqdm(total=EXPERIMENT_TIME, desc=f"profile {payload['id']}", unit="s") as pbar:
            while (now := time.monotonic()) - start < EXPERIMENT_TIME:
                # update the profile if a new one has arrived
                if (update := zmq_link.latest_torque_profile(profiles)) is not None:
                    payload = update
                    torque_profile = zmq_link.config_torque_profile(payload)
                    pbar.set_description(f"profile {payload['id']}")

                # estimate the gait phase and compute the torque for each leg
                position, velocity = read_legs_state(candle)
                smooth_position += LOWPASS_SMOOTHING * (position - smooth_position)
                smooth_velocity += LOWPASS_SMOOTHING * (velocity - smooth_velocity)
                gait_phase = jnp.arctan2(smooth_velocity / gait_omega, smooth_position)
                gain = min((now - start) / GAIN_RAMP_TIME, 1.0)
                torque = gain * TORQUE_SCALE * torque_profile(gait_phase)
                apply_torques(candle, torque)

                # control loop step
                time.sleep(SAMPLING_STEP)
                pbar.n = now - start
                actual_torque = jnp.array([md.getTorque() for md in candle.md80s])
                power = actual_torque * velocity
                pbar.set_postfix(power=f"{float(power.sum()):.3f}")

                # send current timestep measurements back to the driver
                samples.send_json(
                    dict(
                        profile_id=payload["id"],
                        time=now - start,
                        gait_phase0=float(gait_phase[0]),
                        gait_phase1=float(gait_phase[1]),
                        position_0=float(position[0]),
                        position_1=float(position[1]),
                        velocity_0=float(velocity[0]),
                        velocity_1=float(velocity[1]),
                        torque_0=float(torque[0]),
                        torque_1=float(torque[1]),
                        actual_torque_0=float(actual_torque[0]),
                        actual_torque_1=float(actual_torque[1]),
                    )
                )
    finally:
        # always leave the motors unpowered
        apply_torques(candle, jnp.zeros(2))
        candle.end()
