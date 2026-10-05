"""Experiment folder records (one trial_i.json per trial) and the objective."""

import json
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

from exopt.profiles import Profile

BETA = 5.0  # weight of negative (resistive) power in the objective


def penalized_power(samples: dict) -> np.ndarray:
    """Assistance power per sample, averaged over legs, negative power weighted by BETA."""
    power = np.stack(
        [np.asarray(samples[f"torque_{leg}"]) * np.asarray(samples[f"velocity_{leg}"])
         for leg in (0, 1)]
    )
    return np.where(power < 0, BETA * power, power).mean(0)


def objective(samples: dict) -> float:
    """Loss to minimize: minus the mean penalized assistance power."""
    return -float(penalized_power(samples).mean())


class Trial(NamedTuple):
    index: int
    profile: Profile
    samples: dict[str, np.ndarray]

    @property
    def loss(self) -> float:
        return objective(self.samples)


def save(exp_dir: str, index: int, profile: Profile, samples: list[dict]) -> Trial:
    """Record a trial's profile and its samples, stored column-wise."""
    columns = {c: [s[c] for s in samples] for c in samples[0]}
    path = Path(exp_dir) / f"trial_{index}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict() | dict(samples=columns)))
    return Trial(index, profile, {c: np.asarray(v) for c, v in columns.items()})


@lru_cache(maxsize=1024)
def _read(path: Path, mtime: float) -> Trial:
    record = json.loads(path.read_text())
    samples = {c: np.asarray(v) for c, v in record["samples"].items()}
    return Trial(int(path.stem.removeprefix("trial_")), Profile.from_dict(record), samples)


def load(exp_dir: str) -> list[Trial]:
    """All trials in the folder, in index order (cached until a file changes)."""
    paths = Path(exp_dir).glob("trial_*.json")
    trials = [_read(p, p.stat().st_mtime) for p in paths]
    return sorted(trials, key=lambda t: t.index)
