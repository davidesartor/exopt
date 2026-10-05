"""Stand-in for pdo_controller.py: simulate a walking gait and stream samples over ZMQ."""

import time
import jax
import jax.numpy as jnp

from jaxtyping import Array, Float
from tqdm import tqdm
from exopt import MAX_TORQUE, zmq_link

# simulation parameters
SIMULATION_TIME = 60.0  # [s] maximum time to run the simulation
CALIBRATION_TIME = 5.0  # [s] time to calibrate the gait period
SAMPLING_RATE = 100.0  # [Hz] control loop frequency
SAMPLING_STEP = 1.0 / SAMPLING_RATE  # [s] control loop period
LOWPASS_SMOOTHING = 0.5  # EMA weight before the phase-estimation

# subject parameters
TRUE_GAIT_PERIOD = 1.0  # [s] time for one gait cycle (2 steps, 1 per leg)
TRUE_GAIT_OMEGA = 2 * jnp.pi / TRUE_GAIT_PERIOD  # [rad/s] gait angular velocity
LEG_INERTIA = 1.0  # [kg m^2] leg about the hip
HIP_STIFFNESS = 150.0  # [Nm/rad] subject's impedance around the reference gait
HIP_DAMPING = 5.0  # [Nm s/rad]
SUBSTEPS = 10  # integration substeps per control step


@jax.jit
def simulate_legs_state(t: float) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
    """Reference (unassisted) leg positions [rad] and velocities [rad/s] at gait time t."""

    def gait_profile(t: float) -> Float[Array, "2"]:
        # the gait phase for each leg, offset by half a gait cycle
        phase0 = 2 * jnp.pi * t / TRUE_GAIT_PERIOD
        phase = jnp.array([phase0, phase0 + jnp.pi])

        # hip-like trajectory: dominant fundamental plus an asymmetric second harmonic
        return 0.5 * jnp.cos(phase) + 0.1 * jnp.sin(2 * phase + 0.5)

    return jax.jvp(gait_profile, (t,), (1.0,))


@jax.jit
def deviation_step(
    deviation: Float[Array, "2"],
    deviation_rate: Float[Array, "2"],
    torque: Float[Array, "2"],
    dt: float,
) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
    """Advance the exo-induced deviation from the reference gait: I e'' + c e' + k e = torque."""

    def substep(_, state):
        e, de = state
        dde = (torque - HIP_DAMPING * de - HIP_STIFFNESS * e) / LEG_INERTIA
        de = de + dt / SUBSTEPS * dde
        return e + dt / SUBSTEPS * de, de

    return jax.lax.fori_loop(0, SUBSTEPS, substep, (deviation, deviation_rate))


def main():
    # set up ZMQ sockets for streaming samples and receiving profiles
    samples, profiles = zmq_link.controller_link()
    start = time.monotonic()

    # estimate the stride frequency (omega^2 = <velocity^2>/<position^2>)
    with tqdm(total=CALIBRATION_TIME, desc="calibrating gait period", unit="s") as pbar:
        mean_sq_position, mean_sq_velocity = 0.0, 0.0
        gait_omega = None
        while (now := time.monotonic()) - start < CALIBRATION_TIME:
            # accumulate mean squared position and velocity over both legs
            position, velocity = simulate_legs_state(now - start)
            mean_sq_position += float(jnp.sum(position**2))
            mean_sq_velocity += float(jnp.sum(velocity**2))
            gait_omega = jnp.sqrt(mean_sq_velocity / mean_sq_position)

            # simulate a control loop step
            pbar.n = now - start
            pbar.set_postfix(ω=f"{float(gait_omega):.4f} rad/s")
            time.sleep(SAMPLING_STEP)

    # report the estimated gait period and angular velocity
    assert gait_omega is not None, "failed to calibrate gait period"
    gait_period = 2 * jnp.pi / gait_omega
    print(f"ω = {float(gait_omega):.2f} rad/s (true = {TRUE_GAIT_OMEGA:.2f} rad/s)")
    print(f"τ = {float(gait_period):.4f} s (true = {TRUE_GAIT_PERIOD:.2f} s)")

    # block until the first profile arrives
    print(f"Waiting for a profile to stream samples...")
    while (payload := zmq_link.latest_torque_profile(profiles)) is None:
        time.sleep(SAMPLING_STEP)
    torque_profile = zmq_link.config_torque_profile(payload)
    smooth_position, smooth_velocity = simulate_legs_state(time.monotonic() - start)
    deviation = deviation_rate = torque = jnp.zeros(2)
    last_step = time.monotonic()

    # run the simulation, streaming samples and updating the profile when a new one arrives
    with tqdm(total=SIMULATION_TIME, desc=f"profile {payload['id']}", unit="s") as pbar:
        while (now := time.monotonic()) - start < SIMULATION_TIME:
            # the legs respond to the torque applied since the last step (stalls capped)
            dt = min(now - last_step, 5 * SAMPLING_STEP)
            deviation, deviation_rate = deviation_step(
                deviation, deviation_rate, torque, dt
            )
            last_step = now

            # update the profile if a new one has arrived (republished ids are ignored)
            update = zmq_link.latest_torque_profile(profiles)
            if update is not None and update["id"] != payload["id"]:
                payload = update
                torque_profile = zmq_link.config_torque_profile(payload)
                pbar.set_description(f"profile {payload['id']}")

            # estimate the gait phase and compute the torque and power for each leg
            reference_position, reference_velocity = simulate_legs_state(now - start)
            position = reference_position + deviation
            velocity = reference_velocity + deviation_rate
            smooth_position += LOWPASS_SMOOTHING * (position - smooth_position)
            smooth_velocity += LOWPASS_SMOOTHING * (velocity - smooth_velocity)
            gait_phase = jnp.arctan2(-smooth_velocity / gait_omega, smooth_position)
            torque = MAX_TORQUE * jnp.clip(torque_profile(gait_phase), -1.0, 1.0)

            # simulate a control loop step
            time.sleep(SAMPLING_STEP)
            pbar.n = now - start
            power = torque * velocity
            pbar.set_postfix(power=f"{float(power.sum()):.3f}")

            # send current timestep measurements back to the controller
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
                )
            )


if __name__ == "__main__":
    main()
