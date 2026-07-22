import io
import json
import os
import posixpath
import re

import jax.numpy as jnp
import pandas as pd
import paramiko
from jaxtyping import Array, Float

from . import rkhs

# --- Raspberry Pi connection (hardcoded; password overridable via env) ---
PI_HOST = "172.30.207.3"
PI_USER = "pi"
PI_PASSWORD = os.environ.get("PI_PASSWORD", "R0boT21!")

_sftp: paramiko.SFTPClient | None = None


class LocalStorage:
    """Local-filesystem stand-in for the SFTP client.

    Same three calls the rest of this module uses, so nothing downstream knows
    which backend it is talking to. Lets the mock loop run with no hardware
    attached -- and without waiting on a connection to a Pi that is not there.
    """

    def listdir(self, path: str) -> list[str]:
        return os.listdir(path)  # FileNotFoundError is an OSError, as callers expect

    def mkdir(self, path: str) -> None:
        os.mkdir(path)  # FileExistsError is an OSError, as callers expect

    def open(self, path: str, mode: str = "r"):
        return open(path, mode.rstrip("b") + "b")  # callers read/write bytes


def use_local_storage() -> None:
    """Read and write experiment folders on this machine instead of the Pi."""
    global _sftp
    _sftp = LocalStorage()


def _pi():
    """Open (once) and return the storage backend, SFTP to the Pi by default."""
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


def _indices(exp_dir: str, prefix: str) -> dict[int, str]:
    """Map index -> file extension for every {prefix}_{i}.{txt,json} present."""
    try:
        names = _pi().listdir(exp_dir)
    except OSError:  # directory does not exist yet
        return {}
    found = {}
    for name in names:
        m = re.match(rf"{prefix}_(\d+)\.(txt|json)$", name)
        if m:
            found[int(m.group(1))] = m.group(2)
    return found


def config_numbers(exp_dir: str) -> set[int]:
    """Indices of every config file in the folder."""
    return set(_indices(exp_dir, "config"))


def run_numbers(exp_dir: str) -> set[int]:
    """Indices of every run_{i}.txt in the folder."""
    return set(_indices(exp_dir, "run"))


def experiment_mode(exp_dir: str) -> str | None:
    """Infer from the existing configs what kind of variable is optimized here.

    A config is a torque profile if it is JSON (an rkhs.Function) and a
    parameter vector if it is a plain .txt line of floats, so the folder itself
    says which mode to continue in. Returns None for an empty folder.
    """
    extensions = set(_indices(exp_dir, "config").values())
    if not extensions:
        return None
    if extensions == {"json"}:
        return "functional"
    if extensions == {"txt"}:
        return "vector"
    raise ValueError(
        f"{exp_dir} mixes vector (.txt) and functional (.json) configs; "
        "the two are not comparable, so they cannot share an experiment folder."
    )


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


# --- Functional configs: torque profiles over the normalized gait cycle ---
#
# A functional config is a config_{i}.json holding the rkhs.Function itself:
# its own lengthscale rho, its basis points, and its coefficients. That is what
# makes the history reloadable -- a sampled curve alone would not round-trip,
# and the RKHS metric the functional GP is built on needs the exact basis
# expansion and lengthscale, not an interpolant through samples.
#
# The same file also carries a rendered lookup table under "samples", so the
# controller can read the profile directly without knowing anything about the
# RKHS. It is derived from the basis; the basis is the source of truth.
PROFILE_RESOLUTION = 64


def profile_grid(resolution: int = PROFILE_RESOLUTION) -> Float[Array, "m 1"]:
    """Uniform grid of gait phases the torque profile is rendered on."""
    return jnp.linspace(0.0, 1.0, resolution)[:, None]


def save_config_function(
    exp_dir: str, i: int, f: rkhs.Function, resolution: int = PROFILE_RESOLUTION
) -> None:
    """Write an RKHS torque profile as config_{i}.json."""
    grid = profile_grid(resolution)
    payload = dict(
        rho=[float(v) for v in f.rho],
        x=[[float(v) for v in point] for point in f.x],
        a=[float(v) for v in f.a],
        samples=dict(
            phase=[float(v) for v in grid.squeeze(-1)],
            torque=[float(v) for v in f.sample(grid)],
        ),
    )
    _makedirs(exp_dir)
    with _pi().open(posixpath.join(exp_dir, f"config_{i}.json"), "w") as fh:
        fh.write(json.dumps(payload, indent=2).encode())


def read_config_function(exp_dir: str, i: int) -> rkhs.Function:
    """Recover the exact rkhs.Function written to config_{i}.json."""
    with _pi().open(posixpath.join(exp_dir, f"config_{i}.json")) as fh:
        payload = json.loads(fh.read().decode())
    return rkhs.Function(
        rho=jnp.array(payload["rho"]),
        x=jnp.array(payload["x"]),
        a=jnp.array(payload["a"]),
    )


def load_functional_dataset(
    exp_dir: str,
) -> tuple[list[rkhs.Function], Float[Array, "n"]]:
    """Load every profile config that has a matching run, as (fs, ys)."""
    completed = sorted(config_numbers(exp_dir) & run_numbers(exp_dir))
    fs = [read_config_function(exp_dir, i) for i in completed]
    ys = [read_result(exp_dir, i) for i in completed]

    # profiles may differ in lengthscale and in basis size -- that is the point --
    # but they must share an input dimension to be comparable at all
    dims = {f.x.shape[-1] for f in fs}
    if len(dims) > 1:
        raise ValueError(f"{exp_dir} holds profiles of differing input dims: {dims}")

    # the ambient inner product is finite only when rho1^2 + rho2^2 > rho0^2, so
    # nothing in the history may sit below the ambient lengthscale
    rho_min = min((float(f.rho.min()) for f in fs), default=rkhs.RHO_RANGE[0])
    if rho_min < rkhs.RHO_RANGE[0] - 1e-12:
        raise ValueError(
            f"{exp_dir} holds a profile with rho={rho_min:.4g}, below the ambient "
            f"lengthscale {rkhs.RHO_RANGE[0]}; distances would not be finite."
        )
    return fs, jnp.array(ys)


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
