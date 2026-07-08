import glob
import os
import re

import jax.numpy as jnp
import pandas as pd
from jaxtyping import Array, Float


def _indices(exp_dir: str, prefix: str) -> set[int]:
    indices = set()
    for f in glob.glob(os.path.join(exp_dir, f"{prefix}_*.txt")):
        m = re.match(rf"{prefix}_(\d+)\.txt$", os.path.basename(f))
        if m:
            indices.add(int(m.group(1)))
    return indices


def config_numbers(exp_dir: str) -> set[int]:
    """Indices of every config_{i}.txt in the folder."""
    return _indices(exp_dir, "config")


def run_numbers(exp_dir: str) -> set[int]:
    """Indices of every run_{i}.txt in the folder."""
    return _indices(exp_dir, "run")


def next_index(exp_dir: str) -> int:
    """First index not used by any existing config or run file."""
    used = config_numbers(exp_dir) | run_numbers(exp_dir)
    return max(used, default=0) + 1


def read_config(exp_dir: str, i: int) -> Float[Array, "d"]:
    with open(os.path.join(exp_dir, f"config_{i}.txt")) as f:
        return jnp.array([float(v) for v in f.read().split()])


def save_config(exp_dir: str, i: int, x: Float[Array, "d"]) -> None:
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, f"config_{i}.txt"), "w") as f:
        f.write(" ".join(str(float(v)) for v in x))


def loss_function(df: pd.DataFrame, beta: float = 5.0) -> float:
    """Reduce a run dataframe to a scalar objective.

    Area under the mechanicalPower curve for both motors, with regenerative
    (negative) power weighted by beta.
    """
    mechanical_power = df[["mechanicalPower_0", "mechanicalPower_1"]].values
    positive = mechanical_power[mechanical_power > 0]
    negative = mechanical_power[mechanical_power < 0]
    return float((positive.sum() + beta * negative.sum()) / mechanical_power.size)


def save_run(exp_dir: str, i: int, df: pd.DataFrame) -> None:
    """Write a raw hardware-like run dataframe as run_{i}.txt."""
    os.makedirs(exp_dir, exist_ok=True)
    df.to_csv(os.path.join(exp_dir, f"run_{i}.txt"), sep=" ", index=False)


def read_result(exp_dir: str, i: int) -> float:
    """Recover the scalar objective from run_{i}.txt via loss_function."""
    df = pd.read_csv(os.path.join(exp_dir, f"run_{i}.txt"), sep=" ")
    return loss_function(df)


def load_dataset(exp_dir: str) -> tuple[Float[Array, "n d"], Float[Array, "n"]]:
    """Load every config that has a matching run, as (xs, ys) arrays."""
    completed = sorted(config_numbers(exp_dir) & run_numbers(exp_dir))
    xs = [read_config(exp_dir, i) for i in completed]
    ys = [read_result(exp_dir, i) for i in completed]
    return jnp.array(xs), jnp.array(ys)
