from typing import NamedTuple
from functools import lru_cache
from pathlib import Path
import json

import jax.numpy as jnp
import numpy as np
from bofus.rkhs import RBFMixture

BATCHES = 10  # batch means for the standard error of segments with fewer than 2 valid gaits (older trials)
FULL_CYCLE = 0.95 * 2 * np.pi  # phase span of a complete gait, so a cut one (a segment's first) is left out


class Settings(NamedTuple):
    """An experiment's settings, recorded in <exp_dir>/settings.json."""

    atoms: int = 4
    n: int = 4
    min_gaits: int = 10
    max_gaits: int = 20
    beta: float = 5.0
    stop_z: float = 1.96
    min_period: float = 0.5
    max_period: float = 3.0
    min_swing: float = 0.2
    warmup: int = 100
    ramp_cycles: float = 2.0
    min_lengthscale: float = round(float(np.pi * 0.1), 3)
    max_lengthscale: float = round(float(np.pi * np.sqrt(0.6)), 3)
    seed: int = 42
    raw_samples: int = 1024
    max_restarts: int = 32
    power_assist: float = 0.0


DEFAULT = Settings()
SETTINGS_HELP = dict(  # (label, hint) per setting, for the dashboard form and bo-loop --help
    atoms=("Atoms per profile", "RBF bumps summed into one torque profile"),
    n=("Initial design", "random profiles before optimizing"),
    min_gaits=("Min valid gaits per profile", "at least 2, for a standard error"),
    max_gaits=("Max valid gaits per profile", None),
    beta=("Resistive power weight", "negative power counts this many times in the loss"),
    stop_z=("Stop rule z", "a profile stops once its loss - z SE is above the best's mean + z sd"),
    min_period=("Shortest valid gait (s)", None),
    max_period=("Longest valid gait (s)", None),
    min_swing=("Min leg swing (rad)", "position range both legs cover in a valid gait"),
    warmup=("Warm-up samples", "dropped after each ramp up, at 100 Hz"),
    ramp_cycles=("Torque ramp (gait cycles)", "ramp up and down between profiles"),
    min_lengthscale=("Narrowest atom (rad)", "atom lengthscale in gait phase, 2π a cycle"),
    max_lengthscale=("Widest atom (rad)", None),
    seed=("Seed", "initial design and search starts"),
    raw_samples=("Acquisition probes", "random candidates scored before refining"),
    max_restarts=("Acquisition restarts", "candidates refined by gradient ascent"),
    power_assist=("Power assist peak (Nm)", "profiles are offsets from torque in phase with the velocity, this peak"),
)


def save_settings(exp_dir: str, settings: Settings) -> None:
    path = Path(exp_dir) / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings._asdict(), indent=1))


def load_settings(exp_dir: str | None) -> Settings:
    """The folder's settings; defaults for any missing (older experiments have none)."""
    try:
        record = json.loads((Path(exp_dir) / "settings.json").read_text()) if exp_dir else {}
    except (OSError, json.JSONDecodeError):
        record = {}
    return DEFAULT._replace(**{k: type(DEFAULT._asdict()[k])(v) for k, v in record.items() if k in Settings._fields})


def penalized_power(samples: dict, beta: float = DEFAULT.beta) -> np.ndarray:
    """Assistance power per sample, averaged over legs, negative power weighted by beta."""
    power = np.stack(
        [np.asarray(samples[f"torque_{leg}"]) * np.asarray(samples[f"velocity_{leg}"])
         for leg in (0, 1)]
    )
    return np.where(power < 0, beta * power, power).mean(0)


def columns(samples: list[dict]) -> dict[str, np.ndarray]:
    """Numeric sample fields as arrays, missing values as NaN."""
    if not samples:
        return {}
    keys = [k for k in samples[0] if k not in ("stage", "profile_id")]
    return {k: np.array([np.nan if s[k] is None else s[k] for s in samples], dtype=float) for k in keys}


def gait_slices(samples: dict, settings: Settings = DEFAULT) -> list[slice]:
    """The valid gaits: spans between wraps of leg 0's phase covering a full cycle in the period range, legs swinging."""
    phase, time = np.asarray(samples["phase_0"], float), np.asarray(samples["time"], float)
    cuts = [0, *(np.flatnonzero(np.diff(phase) < -np.pi) + 1), len(phase)]

    def valid(gait: slice) -> bool:
        if gait.stop - gait.start < 2:
            return False
        swing = min(np.ptp(np.asarray(samples[f"position_{leg}"][gait], float)) for leg in (0, 1))
        period = time[gait.stop - 1] - time[gait.start]
        return bool(np.ptp(phase[gait]) >= FULL_CYCLE and settings.min_period <= period <= settings.max_period
                    and swing >= settings.min_swing)

    return [gait for gait in map(slice, cuts[:-1], cuts[1:]) if valid(gait)]


def gait_estimate(samples: dict, settings: Settings = DEFAULT) -> tuple[float, float, int]:
    """Loss over the valid gaits (minus their mean penalized power), its SE, and how many there are.

    The SE is the spread of the per-gait losses over sqrt(gaits): gaits are the independent units, samples within one
    are not. Under 2 valid gaits (older trials), loss and SE fall back to all samples, the SE by batch means."""
    power, gaits = penalized_power(samples, settings.beta), gait_slices(samples, settings)
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


def pessimistic_best(rule: dict, z: float) -> float:
    """Upper end of the best observed profile's posterior interval."""
    return rule["best_mean"] + z * rule.get("best_sd", 0.0)


def stop_reason(loss: float, se: float, gaits: int, rule: dict, settings: Settings) -> str | None:
    """Why a segment of `gaits` valid gaits, with this loss and SE, stops now, if it does.

    Between min_gaits and max_gaits it stops in the initial design (no surrogate posterior in the rule), once even its
    optimistic loss is worse than the best's pessimistic posterior mean, or once its SE is below the mean posterior sd
    at the observed profiles.
    """
    if gaits >= settings.max_gaits:
        return "max gaits"
    if gaits < settings.min_gaits:
        return None
    if "best_mean" not in rule:
        return "initial design"
    if loss - settings.stop_z * se > pessimistic_best(rule, settings.stop_z):
        return "worse than best"
    if se < rule["mean_sd"]:
        return "precise enough"
    return None


def encode_profile(profile: RBFMixture) -> dict:
    return {k: v.tolist() for k, v in profile._asdict().items()}


def decode_profile(payload: dict) -> RBFMixture:
    return RBFMixture(*(jnp.asarray(payload[k], dtype=float) for k in RBFMixture._fields))


class Trial(NamedTuple):
    index: int
    profile: RBFMixture
    samples: dict[str, np.ndarray]
    settings: Settings = DEFAULT
    power_assist: float = 0.0

    @property
    def loss(self) -> float:
        """Measured loss: minus the mean penalized power over the valid gaits."""
        return gait_estimate(self.samples, self.settings)[0]

    @property
    def se(self) -> float:
        """Standard error of the measured loss, from gait to gait."""
        return gait_estimate(self.samples, self.settings)[1]


def save(exp_dir: str, index: int, profile: RBFMixture, samples: list[dict], settings: Settings) -> Trial:
    """Record a trial's profile, its power assist, and its samples, stored column-wise."""
    columns = {c: [s[c] for s in samples] for c in samples[0]}
    path = Path(exp_dir) / f"trial_{index}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(encode_profile(profile) | dict(power_assist=settings.power_assist, samples=columns)))
    return Trial(index, profile, {c: np.asarray(v) for c, v in columns.items()}, settings, settings.power_assist)


def save_current(exp_dir: str, payload: dict) -> None:
    """Record the profile being run right now (read by the dashboard)."""
    path = Path(exp_dir) / "current.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def load_current(exp_dir: str) -> dict | None:
    """The profile being run right now, if any."""
    path = Path(exp_dir) / "current.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


@lru_cache(maxsize=1024)
def _read(path: Path, mtime: float, settings: Settings) -> Trial:
    record = json.loads(path.read_text())
    samples = {c: np.asarray(v) for c, v in record["samples"].items()}
    index = int(path.stem.removeprefix("trial"))
    return Trial(index, decode_profile(record), samples, settings, float(record.get("power_assist", 0.0)))


def load(exp_dir: str) -> list[Trial]:
    """All trials in the folder, in index order, under its settings (cached until a file changes)."""
    paths, settings = (Path(exp_dir) / "data").glob("trial*.json"), load_settings(exp_dir)
    trials = [_read(p, p.stat().st_mtime, settings) for p in paths]
    return sorted(trials, key=lambda t: t.index)


