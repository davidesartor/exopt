from typing import NamedTuple
from functools import lru_cache
from pathlib import Path
import json

import numpy as np

from exopt.profiles import Profile

BETA = 5.0  # weight of negative (resistive) power in the objective
BATCHES = 10  # batch means for the standard error of segments with fewer than 2 valid gaits (older trials)
STOP_Z = 1.96  # 95% interval ends: a segment's optimistic loss - STOP_Z SE, the best's pessimistic mean + STOP_Z sd
GAIT_PERIODS = (0.5, 3.0)  # [s] stride periods of a valid gait
MIN_SWING = 0.2  # [rad] position range both legs cover in a valid gait
FULL_CYCLE = 0.95 * 2 * np.pi  # phase span of a complete gait, so a cut one (a segment's first) is left out


def penalized_power(samples: dict) -> np.ndarray:
    """Assistance power per sample, averaged over legs, negative power weighted by BETA."""
    power = np.stack(
        [np.asarray(samples[f"torque_{leg}"]) * np.asarray(samples[f"velocity_{leg}"])
         for leg in (0, 1)]
    )
    return np.where(power < 0, BETA * power, power).mean(0)


def columns(samples: list[dict]) -> dict[str, np.ndarray]:
    """Numeric sample fields as arrays, missing values as NaN."""
    if not samples:
        return {}
    keys = [k for k in samples[0] if k not in ("stage", "profile_id")]
    return {k: np.array([np.nan if s[k] is None else s[k] for s in samples], dtype=float) for k in keys}


def gait_slices(samples: dict) -> list[slice]:
    """The valid gaits: spans between wraps of leg 0's phase covering a full cycle in GAIT_PERIODS, both legs swinging."""
    phase, time = np.asarray(samples["phase_0"], float), np.asarray(samples["time"], float)
    cuts = [0, *(np.flatnonzero(np.diff(phase) < -np.pi) + 1), len(phase)]

    def valid(gait: slice) -> bool:
        if gait.stop - gait.start < 2:
            return False
        swing = min(np.ptp(np.asarray(samples[f"position_{leg}"][gait], float)) for leg in (0, 1))
        period = time[gait.stop - 1] - time[gait.start]
        return bool(np.ptp(phase[gait]) >= FULL_CYCLE and GAIT_PERIODS[0] <= period <= GAIT_PERIODS[1]
                    and swing >= MIN_SWING)

    return [gait for gait in map(slice, cuts[:-1], cuts[1:]) if valid(gait)]


def gait_estimate(samples: dict) -> tuple[float, float, int]:
    """Loss over the valid gaits (minus their mean penalized power), its SE, and how many there are.

    The SE is the spread of the per-gait losses over sqrt(gaits): gaits are the independent units, samples within one
    are not. Under 2 valid gaits (older trials), loss and SE fall back to all samples, the SE by batch means."""
    power, gaits = penalized_power(samples), gait_slices(samples)
    if len(gaits) < 2:
        return -float(power.mean()), standard_error(power), len(gaits)
    per_gait = np.array([-power[gait].mean() for gait in gaits])
    loss = -float(np.concatenate([power[gait] for gait in gaits]).mean())
    return loss, float(per_gait.std(ddof=1) / np.sqrt(len(gaits))), len(gaits)


def standard_error(values: np.ndarray, batches: int = BATCHES) -> float:
    """Standard error of the mean by batch means, robust to the within-cycle correlation."""
    if len(values) < 2 * batches:
        return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
    means = np.array([b.mean() for b in np.array_split(values, batches)])
    return float(means.std(ddof=1) / np.sqrt(batches))


def pessimistic_best(rule: dict) -> float:
    """Upper end of the best observed profile's 95% posterior interval."""
    return rule["best_mean"] + STOP_Z * rule.get("best_sd", 0.0)


def stop_reason(loss: float, se: float, gaits: int, rule: dict) -> str | None:
    """Why a segment of `gaits` valid gaits, with this loss and SE, stops now, if it does.

    Between min_gaits and max_gaits it stops in the initial design (no surrogate in the rule), once even its optimistic
    loss is worse than the best's pessimistic posterior mean, or once its SE is below the mean posterior sd at the
    observed profiles.
    """
    if gaits >= rule["max_gaits"]:
        return "max gaits"
    if gaits < rule["min_gaits"]:
        return None
    if "best_mean" not in rule:
        return "initial design"
    if loss - STOP_Z * se > pessimistic_best(rule):
        return "worse than best"
    if se < rule["mean_sd"]:
        return "precise enough"
    return None


class Trial(NamedTuple):
    index: int
    profile: Profile
    samples: dict[str, np.ndarray]

    @property
    def loss(self) -> float:
        """Measured loss: minus the mean penalized power over the valid gaits."""
        return gait_estimate(self.samples)[0]

    @property
    def se(self) -> float:
        """Standard error of the measured loss, from gait to gait."""
        return gait_estimate(self.samples)[1]


def save(exp_dir: str, index: int, profile: Profile, samples: list[dict]) -> Trial:
    """Record a trial's profile and its samples, stored column-wise."""
    columns = {c: [s[c] for s in samples] for c in samples[0]}
    path = Path(exp_dir) / "data" / f"trial{index}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict() | dict(samples=columns)))
    return Trial(index, profile, {c: np.asarray(v) for c, v in columns.items()})


def save_current(exp_dir: str, payload: dict) -> None:
    """Record the profile being run right now (read by the dashboard)."""
    path = Path(exp_dir) / "data" / "current.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def load_current(exp_dir: str) -> dict | None:
    """The profile being run right now, if any."""
    path = Path(exp_dir) / "data" / "current.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


@lru_cache(maxsize=1024)
def _read(path: Path, mtime: float) -> Trial:
    record = json.loads(path.read_text())
    samples = {c: np.asarray(v) for c, v in record["samples"].items()}
    return Trial(int(path.stem.removeprefix("trial")), Profile.from_dict(record), samples)


def load(exp_dir: str) -> list[Trial]:
    """All trials in the folder, in index order (cached until a file changes)."""
    paths = (Path(exp_dir) / "data").glob("trial*.json")
    trials = [_read(p, p.stat().st_mtime) for p in paths]
    return sorted(trials, key=lambda t: t.index)


