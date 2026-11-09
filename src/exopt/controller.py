"""Controller script: estimate the gait phase, apply the published torque profile to the drives, stream samples."""

import signal
import sys
import time
from typing import NamedTuple

import numpy as np
from jaxtyping import Bool, Float

from exopt import MAX_TORQUE, link
from exopt.profiles import Profile

SAMPLING_STEP = 0.01  # [s] control loop period
MIN_CALIBRATION_TIME = 10.0  # [s] minimum time to calibrate the gait frequency
CALIBRATION_TOL = 0.01  # relative drift of the stride frequency per second at convergence
AFO_HARMONICS = 4  # harmonics of the trajectory the oscillator learns
AFO_BANDWIDTH = 0.5  # [rad/s] natural frequency of the oscillator's phase-locking loop, damping 0.7
AFO_SHAPE_TIME = 5.0  # [s] time constant of the learned trajectory
AFO_SETTLE_TIME = 5.0  # [s] the oscillator locks after calibration before torque is allowed
MAX_OMEGA_SLIP = 0.25  # log ratio of oscillator to stride frequency beyond which omega resets to the stride one
RECENT_TIME = 3.0  # [s] EMA time constant of the recent position mean and spread
CROSSING_HYSTERESIS = 0.3  # zero-crossing band, as a fraction of the recent position RMS
MIN_CROSSING_BAND = 0.03  # [rad] floor of the band, so standing noise never counts as strides
STRIDES_KEPT = 6  # recent stride periods whose median sets the stride frequency
WALKING_STRIDES = 2.0  # [strides] without a crossing on either leg before the oscillators stop adapting
STILL_SPEED = 0.15  # [rad/s] both legs slower than this count as standing still (velocity noise sd 0.03)
STILL_TIME = 1.0  # [s] standing still before the logged positions are re-zeroed on the standing pose
IDLE_TIMEOUT = 300.0  # [s] with nothing connected before the controller exits
GAIN_RAMP_CYCLES = 3.0  # [gait cycles] ramp of the profile gain between zero and full scale


class Estimator(NamedTuple):
    """Gait phase from an adaptive frequency oscillator per leg, sharing one omega, guarded by a stride timer.

    Each oscillator learns its leg's trajectory as a Fourier series in its phase, the fundamental pinned to a
    cosine (x ~ cos(phase)). Strides timed by zero crossings start the oscillators, pause their adaptation
    while the subject stands, and reset omega if it slips away (lost lock, subharmonic).
    """

    # stride timer
    started_at: float | None = None
    recent_mean: Float[np.ndarray, "2"] | None = None
    recent_sq: Float[np.ndarray, "2"] = np.zeros(2)
    below: Bool[np.ndarray, "2"] = np.zeros(2, bool)
    rising_at: tuple[tuple[float, ...], tuple[float, ...]] = ((), ())
    stride_omega: float | None = None
    walking: bool = False
    resumed: bool = False

    # calibration
    checked_at: float | None = None
    checked_omega: float | None = None

    # oscillators, once calibrated
    locked_at: float | None = None
    omega: float | None = None
    phase: Float[np.ndarray, "2"] | None = None
    cos_coefficients: Float[np.ndarray, "2 H"] | None = None
    sin_coefficients: Float[np.ndarray, "2 H"] | None = None


def time_strides(estimator: Estimator, now: float, position: Float[np.ndarray, "2"]) -> Estimator:
    """Stride frequency from the median period between rising zero crossings around the recent mean."""
    if estimator.recent_mean is None:
        estimator = estimator._replace(started_at=now, checked_at=now, recent_mean=position)
    recent = SAMPLING_STEP / RECENT_TIME
    recent_mean = estimator.recent_mean + recent * (position - estimator.recent_mean)
    recent_sq = estimator.recent_sq + recent * ((position - recent_mean) ** 2 - estimator.recent_sq)

    # rising crossings with hysteresis keep the fundamental, unlike <v^2>/<x^2>
    centered = position - recent_mean
    band = max(CROSSING_HYSTERESIS * float(np.mean(recent_sq)) ** 0.5, MIN_CROSSING_BAND)
    rising = estimator.below & (centered > band)
    below = (centered < -band) | (estimator.below & ~rising)
    rising_at = tuple(
        (times + (now,))[-STRIDES_KEPT - 1:] if rose else times for times, rose in zip(estimator.rising_at, rising)
    )

    stride_omega = estimator.stride_omega
    if all(len(times) > STRIDES_KEPT for times in rising_at):
        stride_omega = 2 * np.pi / np.median([np.diff(times) for times in rising_at])
    walking = stride_omega is not None and all(
        now - times[-1] < WALKING_STRIDES * 2 * np.pi / stride_omega for times in rising_at
    )
    return estimator._replace(
        recent_mean=recent_mean, recent_sq=recent_sq, below=below, rising_at=rising_at,
        stride_omega=stride_omega, walking=walking, resumed=walking and not estimator.walking,
    )


def calibrate(
    estimator: Estimator, now: float, position: Float[np.ndarray, "2"], velocity: Float[np.ndarray, "2"]
) -> Estimator:
    """Once the stride frequency drifts < tol per second, start the oscillators on the recent sinusoid."""
    if now - estimator.checked_at < 1.0:
        return estimator
    settled = estimator.stride_omega is not None and estimator.checked_omega is not None
    drift = abs(estimator.stride_omega - estimator.checked_omega) / estimator.stride_omega if settled else np.inf
    estimator = estimator._replace(checked_omega=estimator.stride_omega, checked_at=now)
    if now - estimator.started_at < MIN_CALIBRATION_TIME or drift >= CALIBRATION_TOL:
        return estimator

    omega = estimator.stride_omega
    fundamental = np.sqrt(2 * estimator.recent_sq)
    return estimator._replace(
        locked_at=now,
        omega=omega,
        phase=np.arctan2(-velocity / omega, position - estimator.recent_mean),
        cos_coefficients=np.column_stack([estimator.recent_mean, fundamental, np.zeros((2, AFO_HARMONICS - 1))]),
        sin_coefficients=np.zeros((2, AFO_HARMONICS + 1)),
    )


def advance_oscillators(
    estimator: Estimator, position: Float[np.ndarray, "2"], velocity: Float[np.ndarray, "2"]
) -> Estimator:
    """Advance the oscillators one step; they adapt only while walking, and run free otherwise."""
    cos_coefficients, sin_coefficients = estimator.cos_coefficients, estimator.sin_coefficients

    # back from a stop, restart the phase on the learned walking mean, as at calibration
    phase = estimator.phase
    if estimator.resumed:
        phase = np.arctan2(-velocity / estimator.omega, position - cos_coefficients[:, 0])
    omega = estimator.omega
    if estimator.walking and abs(np.log(omega / estimator.stride_omega)) > MAX_OMEGA_SLIP:
        omega = estimator.stride_omega

    harmonics = np.arange(AFO_HARMONICS + 1)
    cos, sin = np.cos(np.outer(phase, harmonics)), np.sin(np.outer(phase, harmonics))
    error = estimator.walking * (position - np.sum(cos_coefficients * cos + sin_coefficients * sin, axis=1))

    # phase-locking loop: error along the trajectory's slope, normalized so the loop gain ignores the amplitude
    slope = np.sum(harmonics * (sin_coefficients * cos - cos_coefficients * sin), axis=1)
    mean_sq_slope = 0.5 * np.sum(harmonics**2 * (cos_coefficients**2 + sin_coefficients**2), axis=1)
    phase_error = error * slope / np.maximum(mean_sq_slope, MIN_CROSSING_BAND**2)
    phase = np.angle(np.exp(1j * (phase + SAMPLING_STEP * (omega + 1.4 * AFO_BANDWIDTH * phase_error))))
    omega = omega + SAMPLING_STEP * AFO_BANDWIDTH**2 * float(np.mean(phase_error))

    # learn the trajectory by gradient steps on the squared error; the fundamental stays a positive cosine
    learning_rate = 2 * SAMPLING_STEP / AFO_SHAPE_TIME
    cos_coefficients = cos_coefficients + learning_rate * error[:, None] * cos
    sin_coefficients = sin_coefficients + learning_rate * error[:, None] * sin
    cos_coefficients[:, 1] = np.maximum(cos_coefficients[:, 1], MIN_CROSSING_BAND)
    sin_coefficients[:, 1] = 0.0
    return estimator._replace(
        phase=phase, omega=omega, cos_coefficients=cos_coefficients, sin_coefficients=sin_coefficients
    )


def estimate_phase(
    estimator: Estimator, now: float, position: Float[np.ndarray, "2"], velocity: Float[np.ndarray, "2"]
) -> tuple[Estimator, Float[np.ndarray, "2"] | None]:
    """Phase [rad] of both legs, or None while calibrating and settling."""
    estimator = time_strides(estimator, now, position)
    if estimator.omega is None:
        return calibrate(estimator, now, position, velocity), None
    estimator = advance_oscillators(estimator, position, velocity)
    return estimator, estimator.phase if now - estimator.locked_at >= AFO_SETTLE_TIME else None


if __name__ == "__main__":
    import pyCandle

    # two MD80 drives over CAN in raw torque mode
    candle = pyCandle.Candle(pyCandle.CAN_BAUD_1M, True)
    ids = candle.ping()[:2]
    if len(ids) < 2:
        sys.exit(f"expected 2 motors, found {len(ids)}")
    for id in ids:
        candle.addMd80(id)
    for id, md in zip(ids, candle.md80s):
        if not candle.controlMd80Mode(id, pyCandle.RAW_TORQUE):
            sys.exit(f"failed to set mode on drive {id}")
        if not candle.controlMd80Enable(id, True):
            sys.exit(f"failed to enable drive {id}")
        md.setMaxTorque(MAX_TORQUE)
    candle.begin()

    # a dropped ssh session ends the run like ctrl-c, so the drives are released
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.default_int_handler)
    samples, profiles = link.controller_link()
    peers = link.Peers(samples, profiles)

    estimator = Estimator()
    # every profile ramps up from zero and back down to zero, so each trial starts alike
    requested, applied_id, gain = dict(id=None), None, 0.0
    harmonics, cos, sin = np.arange(1), np.zeros(1), np.zeros(1)
    start = last_tick = next_tick = idle_since = still_since = time.monotonic()
    standing_pose = None
    stage = None

    # stream samples and apply the requested profiles until ctrl-c, or IDLE_TIMEOUT with nothing connected
    try:
        while True:
            now = time.monotonic()
            if peers():
                idle_since = now
            elif now - idle_since > IDLE_TIMEOUT:
                print(f"nothing connected for {IDLE_TIMEOUT / 60:g} min: exiting")
                break
            raw_position = np.array([md.getPosition() for md in candle.md80s])
            velocity = np.array([md.getVelocity() for md in candle.md80s])
            measured_torque = np.array([md.getTorque() for md in candle.md80s])

            # logged positions are zeroed on the standing pose, re-taken whenever the subject stands still
            if np.any(np.abs(velocity) > STILL_SPEED):
                still_since = now
            if standing_pose is None or now - still_since >= STILL_TIME:
                standing_pose = raw_position
            position = raw_position - standing_pose

            # calibration starts by itself once strides show up; a stop ramps the torque down until walking resumes
            estimator, phase = estimate_phase(estimator, now, raw_position, velocity)
            previous_stage = stage
            stage = "standing" if not estimator.walking else "running" if phase is not None else "calibrating"
            if stage != previous_stage:
                print(f"running: gait period {2 * np.pi / estimator.omega:.3f} s" if stage == "running" else stage)

            # a request is a profile, or a release (id None); republished ids are ignored; none before calibration
            update = link.latest_message(profiles) if phase is not None else None
            if update is not None and update["id"] != requested["id"]:
                requested = update
                print(f"profile {requested['id']}" if requested["id"] else "release")

            # swap profiles only at zero gain, then ramp up; ramp down on release, a new request or a stop
            if gain == 0.0 and requested["id"] not in (None, applied_id):
                applied_id = requested["id"]
                cos, sin = (np.asarray(c) for c in Profile.from_dict(requested).series())
                harmonics = np.arange(len(cos))
            applying = requested["id"] is not None and requested["id"] == applied_id and stage == "running"
            target = 1.0 if applying else 0.0
            step = (now - last_tick) * estimator.omega / (2 * np.pi * GAIN_RAMP_CYCLES) if phase is not None else 0.0
            gain = min(gain + step, target) if gain < target else max(gain - step, target)
            last_tick = now

            # zero torque until calibrated
            torque = np.zeros(2)
            if phase is not None:
                angle = np.outer(phase, harmonics)
                torque = MAX_TORQUE * np.clip(gain * (np.cos(angle) @ cos + np.sin(angle) @ sin), -1.0, 1.0)
            for md, leg_torque in zip(candle.md80s, torque):
                md.setTargetTorque(float(leg_torque))

            columns = dict(phase=phase, position=position, velocity=velocity, torque=torque,
                           measured_torque=measured_torque)
            samples.send_json(
                dict(time=now - start, stage=stage, profile_id=applied_id, gain=gain)
                | {f"{name}_{leg}": None if values is None else float(values[leg])
                   for name, values in columns.items() for leg in (0, 1)}
            )
            next_tick = max(next_tick + SAMPLING_STEP, time.monotonic())
            time.sleep(max(next_tick - time.monotonic(), 0.0))

    # leave the motors unpowered, however the run ends
    finally:
        for md in candle.md80s:
            md.setTargetTorque(0.0)
        candle.end()
        for id in ids:
            candle.controlMd80Enable(id, False)
