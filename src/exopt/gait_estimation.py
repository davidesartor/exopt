"""Gait phase of both legs: adaptive frequency oscillators, started and guarded by a stride timer."""

import numpy as np
from jaxtyping import Bool, Float

HARMONICS = 4  # harmonics of the trajectory an oscillator learns
BANDWIDTH = 0.5  # [rad/s] natural frequency of the oscillators' phase-locking loop, damping 0.7
SHAPE_TIME = 5.0  # [s] time constant of the learned trajectory
SETTLE_TIME = 5.0  # [s] the oscillators lock after calibration before their phase is given out
MAX_OMEGA_SLIP = 0.25  # log ratio of oscillator to stride frequency beyond which omega resets to the stride one
MIN_CALIBRATION_TIME = 10.0  # [s] minimum time to calibrate the gait frequency
CALIBRATION_TOL = 0.01  # relative drift of the stride frequency per second at convergence
RECENT_TIME = 3.0  # [s] EMA time constant of the recent position mean and spread
CROSSING_HYSTERESIS = 0.3  # zero-crossing band, as a fraction of the recent position RMS
MIN_CROSSING_BAND = 0.03  # [rad] floor of the band, so standing noise never counts as strides
STRIDES_KEPT = 6  # recent stride periods whose median sets the stride frequency
WALKING_STRIDES = 2.0  # [strides] without a crossing on either leg before the subject counts as standing


def sinusoid_phase(
    centered: Float[np.ndarray, "2"], velocity: Float[np.ndarray, "2"], omega: float
) -> Float[np.ndarray, "2"]:
    """Phase [rad] of legs moving as x ~ cos(phase) around zero."""
    return np.arctan2(-velocity / omega, centered)


class StrideTimer:
    """Strides timed by each leg's rising zero crossings around its recent mean position."""

    def __init__(self, step: float):
        self.step = step
        self.now = 0.0
        self.mean: Float[np.ndarray, "2"] | None = None
        self.mean_sq: Float[np.ndarray, "2"] = np.zeros(2)
        self.below: Bool[np.ndarray, "2"] = np.zeros(2, bool)
        self.rising_at: tuple[list[float], list[float]] = ([], [])

    @property
    def band(self) -> float:
        """Hysteresis [rad] a crossing must clear: with it the crossings keep the fundamental, unlike <v^2>/<x^2>."""
        return max(CROSSING_HYSTERESIS * float(np.mean(self.mean_sq)) ** 0.5, MIN_CROSSING_BAND)

    @property
    def amplitude(self) -> Float[np.ndarray, "2"]:
        """Amplitude [rad] of the sinusoid with the recent spread."""
        return np.sqrt(2 * self.mean_sq)

    @property
    def omega(self) -> float | None:
        """Stride frequency [rad/s] from the median of the recent periods, once both legs have STRIDES_KEPT."""
        if any(len(times) <= STRIDES_KEPT for times in self.rising_at):
            return None
        return float(2 * np.pi / np.median([np.diff(times) for times in self.rising_at]))

    @property
    def walking(self) -> bool:
        """Both legs crossed within the last WALKING_STRIDES."""
        omega = self.omega
        return omega is not None and all(
            self.now - times[-1] < WALKING_STRIDES * 2 * np.pi / omega for times in self.rising_at)

    def update(self, now: float, position: Float[np.ndarray, "2"]) -> None:
        self.now = now
        recent = self.step / RECENT_TIME
        self.mean = position if self.mean is None else self.mean + recent * (position - self.mean)
        centered = position - self.mean
        self.mean_sq = self.mean_sq + recent * (centered**2 - self.mean_sq)

        band = self.band
        rising = self.below & (centered > band)
        self.below = (centered < -band) | (self.below & ~rising)
        for times, rose in zip(self.rising_at, rising):
            if rose:
                times.append(now)
                del times[: -STRIDES_KEPT - 1]


class Oscillators:
    """One oscillator per leg sharing omega, each learning its leg's trajectory as a Fourier series in its phase.

    The fundamental is pinned to a positive cosine, so x ~ cos(phase)."""

    def __init__(self, step: float, omega: float, phase: Float[np.ndarray, "2"], mean: Float[np.ndarray, "2"],
                 amplitude: Float[np.ndarray, "2"]):
        self.step, self.omega, self.phase = step, omega, phase
        self.cos: Float[np.ndarray, "2 H"] = np.column_stack([mean, amplitude, np.zeros((2, HARMONICS - 1))])
        self.sin: Float[np.ndarray, "2 H"] = np.zeros((2, HARMONICS + 1))

    @property
    def mean(self) -> Float[np.ndarray, "2"]:
        """Learned mean position [rad] of the walking trajectory."""
        return self.cos[:, 0]

    def advance(self, position: Float[np.ndarray, "2"], adapt: bool) -> None:
        """One step; without `adapt` the oscillators run free."""
        harmonics = np.arange(HARMONICS + 1)
        cos, sin = np.cos(np.outer(self.phase, harmonics)), np.sin(np.outer(self.phase, harmonics))
        error = adapt * (position - np.sum(self.cos * cos + self.sin * sin, axis=1))

        # phase-locking loop: error along the trajectory's slope, normalized so the loop gain ignores the amplitude
        slope = np.sum(harmonics * (self.sin * cos - self.cos * sin), axis=1)
        mean_sq_slope = 0.5 * np.sum(harmonics**2 * (self.cos**2 + self.sin**2), axis=1)
        phase_error = error * slope / np.maximum(mean_sq_slope, MIN_CROSSING_BAND**2)
        self.phase = np.angle(np.exp(1j * (self.phase + self.step * (self.omega + 1.4 * BANDWIDTH * phase_error))))
        self.omega += self.step * BANDWIDTH**2 * float(np.mean(phase_error))

        # learn the trajectory by gradient steps on the squared error; the fundamental stays a positive cosine
        learning_rate = 2 * self.step / SHAPE_TIME
        self.cos = self.cos + learning_rate * error[:, None] * cos
        self.sin = self.sin + learning_rate * error[:, None] * sin
        self.cos[:, 1] = np.maximum(self.cos[:, 1], MIN_CROSSING_BAND)
        self.sin[:, 1] = 0.0


class PhaseEstimator:
    """Gait phase [rad] of both legs from positions and velocities sampled every `step` seconds.

    The stride timer starts the oscillators once its frequency settles, pauses their adaptation while the
    subject stands, and resets their omega if it slips away (lost lock, subharmonic)."""

    def __init__(self, step: float):
        self.step = step
        self.strides = StrideTimer(step)
        self.oscillators: Oscillators | None = None
        self.started_at: float | None = None
        self.locked_at: float | None = None
        self.checked_at = 0.0
        self.checked_omega: float | None = None

    @property
    def walking(self) -> bool:
        return self.strides.walking

    @property
    def omega(self) -> float | None:
        """Gait frequency [rad/s] of the oscillators, None until calibrated."""
        return None if self.oscillators is None else self.oscillators.omega

    @property
    def phase(self) -> Float[np.ndarray, "2"] | None:
        """None while calibrating and settling."""
        if self.oscillators is None or self.strides.now - self.locked_at < SETTLE_TIME:
            return None
        return self.oscillators.phase

    def update(self, now: float, position: Float[np.ndarray, "2"], velocity: Float[np.ndarray, "2"]) -> None:
        if self.started_at is None:
            self.started_at = self.checked_at = now
        was_walking = self.walking
        self.strides.update(now, position)
        if self.oscillators is None:
            self.calibrate(now, position, velocity)
            return

        # back from a stop, restart the phase on the learned walking mean, as at calibration
        if self.walking and not was_walking:
            self.oscillators.phase = sinusoid_phase(position - self.oscillators.mean, velocity, self.oscillators.omega)
        if self.walking and abs(np.log(self.oscillators.omega / self.strides.omega)) > MAX_OMEGA_SLIP:
            self.oscillators.omega = self.strides.omega
        self.oscillators.advance(position, adapt=self.walking)

    def calibrate(self, now: float, position: Float[np.ndarray, "2"], velocity: Float[np.ndarray, "2"]) -> None:
        """Once the stride frequency drifts < tol per second, start the oscillators on the recent sinusoid."""
        if now - self.checked_at < 1.0:
            return
        omega, previous = self.strides.omega, self.checked_omega
        self.checked_at, self.checked_omega = now, omega
        if omega is None or previous is None or now - self.started_at < MIN_CALIBRATION_TIME:
            return
        if abs(omega - previous) / omega < CALIBRATION_TOL:
            phase = sinusoid_phase(position - self.strides.mean, velocity, omega)
            self.oscillators = Oscillators(self.step, omega, phase, self.strides.mean, self.strides.amplitude)
            self.locked_at = now
