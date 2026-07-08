import io
import os
import posixpath
import re

import jax.numpy as jnp
import pandas as pd
import paramiko
from jaxtyping import Array, Float

# --- Raspberry Pi connection (hardcoded; password overridable via env) ---
PI_HOST = "172.30.207.3"
PI_USER = "pi"
PI_PASSWORD = os.environ.get("PI_PASSWORD", "R0boT21!")

_sftp: paramiko.SFTPClient | None = None


def _pi() -> paramiko.SFTPClient:
    """Open (once) and return the SFTP client to the Pi."""
    global _sftp
    if _sftp is None:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(PI_HOST, username=PI_USER, password=PI_PASSWORD)
        _sftp = client.open_sftp()
    return _sftp


def _makedirs(exp_dir: str) -> None:
    """Recursive mkdir over SFTP (sftp.mkdir is not recursive)."""
    sftp = _pi()
    parents = []
    p = exp_dir.rstrip("/")
    while p and p not in ("/", "."):
        parents.append(p)
        p = posixpath.dirname(p)
    for d in reversed(parents):
        try:
            sftp.mkdir(d)
        except OSError:  # already exists
            pass


def _indices(exp_dir: str, prefix: str) -> set[int]:
    try:
        names = _pi().listdir(exp_dir)
    except OSError:  # directory does not exist yet
        return set()
    indices = set()
    for name in names:
        m = re.match(rf"{prefix}_(\d+)\.txt$", name)
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
    with _pi().open(posixpath.join(exp_dir, f"config_{i}.txt")) as f:
        text = f.read().decode()
    return jnp.array([float(v) for v in text.split()])


def save_config(exp_dir: str, i: int, x: Float[Array, "d"]) -> None:
    _makedirs(exp_dir)
    with _pi().open(posixpath.join(exp_dir, f"config_{i}.txt"), "w") as f:
        f.write(" ".join(str(float(v)) for v in x).encode())


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
    _makedirs(exp_dir)
    with _pi().open(posixpath.join(exp_dir, f"run_{i}.txt"), "w") as f:
        f.write(df.to_csv(sep=" ", index=False).encode())


def read_result(exp_dir: str, i: int) -> float:
    """Recover the scalar objective from run_{i}.txt via loss_function."""
    with _pi().open(posixpath.join(exp_dir, f"run_{i}.txt")) as f:
        text = f.read().decode()
    df = pd.read_csv(io.StringIO(text), sep=" ")
    return loss_function(df)


def load_dataset(exp_dir: str) -> tuple[Float[Array, "n d"], Float[Array, "n"]]:
    """Load every config that has a matching run, as (xs, ys) arrays."""
    completed = sorted(config_numbers(exp_dir) & run_numbers(exp_dir))
    xs = [read_config(exp_dir, i) for i in completed]
    ys = [read_result(exp_dir, i) for i in completed]
    return jnp.array(xs), jnp.array(ys)
