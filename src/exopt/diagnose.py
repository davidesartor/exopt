"""Drive check: both drives answer, then apply a 1 Nm periodic gait torque until stopped, reporting tracking."""

import signal
import time

import numpy as np
from jaxtyping import Float

from exopt import streams
from exopt.control_loop import SAMPLING_STEP
from exopt.drives_md80 import Drives

GAIT_PERIOD = 1.1  # [s] of the applied torque, legs half a period apart
AMPLITUDE = 1.0  # [Nm] of the gait torque, ramped into over the first cycle
REPORT_CYCLES = 4  # tracking is reported over each stretch of this many cycles
MIN_GAIN, MAX_GAIN = 0.8, 1.2  # measured over commanded torque of a working drive
BRAKE_GAIN = 0.2  # below this the drive ignores the command: off (watchdog) or faulted
VIEWER_WAIT = 10.0  # [s] for the dashboard to connect, so its live view shows the whole run
STALL_REPORTED = 0.05  # [s] of identical reads worth reporting; the MD80 watchdog trips at 0.2-0.25 s


def fit_tracking(
    commanded: Float[np.ndarray, "T"], measured: Float[np.ndarray, "T"], velocity: Float[np.ndarray, "T"]
) -> tuple[float, float, float]:
    """measured = gain * commanded + damping * velocity + offset: the gain, the damping and the rms residual."""
    design = np.stack([commanded, velocity, np.ones_like(commanded)], axis=-1)
    coefficients, *_ = np.linalg.lstsq(design, measured, rcond=None)
    residual = measured - design @ coefficients
    return float(coefficients[0]), float(coefficients[1]), float(np.sqrt(np.mean(residual**2)))


def verdict(gain: float, damping: float) -> str:
    if MIN_GAIN <= gain <= MAX_GAIN:
        return "ok"
    if abs(gain) < BRAKE_GAIN:
        return "NOT APPLYING" + (" (passive brake: drive off, e.g. CAN watchdog)" if damping < -1.0 else "")
    return "POOR TRACKING"


def longest_stall(time_s: Float[np.ndarray, "T"], reads: Float[np.ndarray, "T K"]) -> float:
    """Longest stretch [s] over which every read repeated exactly (the CAN link stopped updating)."""
    changed = np.flatnonzero(np.any(np.diff(reads, axis=0) != 0, axis=-1))
    change_times = np.concatenate([time_s[:1], time_s[changed + 1], time_s[-1:]])
    return float(np.diff(change_times).max(initial=0.0))


if __name__ == "__main__":
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.default_int_handler)

    with Drives() as drives:
        # samples stream to the dashboard's live view as the controller's do
        samples, profiles = streams.controller_link()
        peers, waited_since = streams.Peers(samples, profiles), time.monotonic()
        print("waiting for the dashboard to connect...")
        while not peers() and time.monotonic() - waited_since < VIEWER_WAIT:
            time.sleep(0.1)
        print("dashboard connected" if peers() else "no dashboard connected: running without the live view")

        # apply the gait torque until ctrl-c, streaming samples and reporting each stretch's tracking
        print(f"applying a {AMPLITUDE:g} Nm, {GAIT_PERIOD} s gait torque, legs in antiphase, until stopped")
        stretch, start = [], time.monotonic()
        next_tick, next_report, longest = start, REPORT_CYCLES * GAIT_PERIOD, 0.0
        try:
            while True:
                t = time.monotonic() - start
                position, velocity, measured = drives.read(t)

                gain = min(t / GAIT_PERIOD, 1.0)
                phase = (2 * np.pi * t / GAIT_PERIOD + np.pi * np.arange(2)) % (2 * np.pi)
                commanded = gain * AMPLITUDE * np.sin(phase)
                drives.apply(commanded)
                stretch.append(np.concatenate([[t], commanded, measured, velocity, position]))
                columns = dict(phase=phase, position=position, velocity=velocity, torque=commanded,
                               measured_torque=measured)
                samples.send_json(
                    dict(time=t, stage="running", profile_id=None, gain=gain)
                    | {f"{name}_{leg}": float(values[leg]) for name, values in columns.items() for leg in (0, 1)}
                )

                if t >= next_report:
                    log = np.array(stretch)
                    for leg, id in enumerate(drives.ids):
                        # a drive silent on the bus leaves pyCandle repeating its last reads
                        stall = longest_stall(log[:, 0], log[:, [3 + leg, 5 + leg, 7 + leg]])
                        longest = max(longest, stall)
                        fit_gain, damping, residual = fit_tracking(log[:, 1 + leg], log[:, 3 + leg], log[:, 5 + leg])
                        outcome = verdict(fit_gain, damping)
                        if stall >= 0.9 * (log[-1, 0] - log[0, 0]):
                            outcome = "NO FRESH READS (drive silent on the CAN bus, values are the last ones it sent)"
                        elif stall >= STALL_REPORTED:
                            outcome += f", reads frozen for {stall:.2f} s"
                        print(f"t={t:5.0f} s drive {id}: measured = {fit_gain:+.2f} x commanded {damping:+.2f} "
                              f"x velocity (rms {residual:.2f} Nm), moved {np.ptp(log[:, 7 + leg]):.3f} rad: {outcome}")
                    stretch, next_report = [], next_report + REPORT_CYCLES * GAIT_PERIOD

                next_tick = max(next_tick + SAMPLING_STEP, time.monotonic())
                time.sleep(max(next_tick - time.monotonic(), 0.0))
        except KeyboardInterrupt:
            print(f"stopped: longest read freeze {longest:.2f} s")
