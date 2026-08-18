"""Experiment folder records: config_i.json / run_i.txt pairs."""

import io
import json
import os
from pathlib import Path

import numpy as np

from exopt import acquisition


def config_numbers(exp_dir: str) -> set[int]:
    return {int(p.stem.removeprefix("config_")) for p in Path(exp_dir).glob("config_*.json")}


def run_numbers(exp_dir: str) -> set[int]:
    return {int(p.stem.removeprefix("run_")) for p in Path(exp_dir).glob("run_*.txt")}


def next_index(exp_dir: str) -> int:
    return max(config_numbers(exp_dir) | run_numbers(exp_dir), default=0) + 1


def save_config(exp_dir: str, i: int, payload: dict) -> None:
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, f"config_{i}.json"), "w") as fh:
        json.dump(payload, fh, indent=2)


def read_config(exp_dir: str, i: int) -> dict:
    with open(os.path.join(exp_dir, f"config_{i}.json")) as fh:
        return json.load(fh)


def newest_mode(exp_dir: str) -> str | None:
    """'vector' if the newest config is a single h=1 atom, else 'functional'."""
    indices = config_numbers(exp_dir)
    if not indices:
        return None
    cfg = read_config(exp_dir, max(indices))
    is_vector = cfg["harmonics"] == 1 and len(cfg["amplitudes"]) == 1
    return "vector" if is_vector else "functional"


def write_run(exp_dir: str, i: int, collected: list[dict]) -> None:
    columns = list(collected[0])
    header = " ".join(columns)
    rows = [" ".join(str(s[c]) for c in columns) for s in collected]
    with open(os.path.join(exp_dir, f"run_{i}.txt"), "w") as fh:
        fh.write(header + "\n" + "\n".join(rows) + "\n")


def read_result(exp_dir: str, i: int) -> float:
    with open(os.path.join(exp_dir, f"run_{i}.txt")) as fh:
        trace = np.genfromtxt(io.StringIO(fh.read()), names=True)
    return acquisition.loss_function(trace)
