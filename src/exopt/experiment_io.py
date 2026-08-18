"""Experiment folder records: config_i.json / run_i.txt pairs."""

import io
import json
import os
import numpy as np

from pathlib import Path
from exopt import acquisition


def config_numbers(exp_dir: str) -> set[int]:
    return {int(p.stem.removeprefix("config_")) for p in Path(exp_dir).glob("config_*.json")}


def run_numbers(exp_dir: str) -> set[int]:
    return {int(p.stem.removeprefix("run_")) for p in Path(exp_dir).glob("run_*.txt")}


def next_index(exp_dir: str) -> int:
    """First index after every config or run already in the folder."""
    return max(config_numbers(exp_dir) | run_numbers(exp_dir), default=0) + 1


def save_config(exp_dir: str, i: int, payload: dict) -> None:
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, f"config_{i}.json"), "w") as fh:
        json.dump(payload, fh, indent=2)


def read_config(exp_dir: str, i: int) -> dict:
    with open(os.path.join(exp_dir, f"config_{i}.json")) as fh:
        return json.load(fh)


def newest_mode(exp_dir: str) -> str | None:
    """'vector' if the newest config has a single harmonic, else 'functional'."""
    indices = config_numbers(exp_dir)
    if not indices:
        return None
    cfg = read_config(exp_dir, max(indices))
    return "vector" if len(cfg["sin"]) == 1 else "functional"


def write_run(exp_dir: str, i: int, collected: list[dict]) -> None:
    """Write a segment of samples as a whitespace table with a header row."""
    columns = list(collected[0])
    header = " ".join(columns)
    rows = [" ".join(str(s[c]) for c in columns) for s in collected]
    with open(os.path.join(exp_dir, f"run_{i}.txt"), "w") as fh:
        fh.write(header + "\n" + "\n".join(rows) + "\n")


def read_result(exp_dir: str, i: int) -> float:
    """Loss of a recorded run, recomputed from its sample trace."""
    with open(os.path.join(exp_dir, f"run_{i}.txt")) as fh:
        trace = np.genfromtxt(io.StringIO(fh.read()), names=True)
    return acquisition.objective(trace)
