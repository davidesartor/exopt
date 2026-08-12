import io
import json
import os
import posixpath
import re

import jax.numpy as jnp
import pandas as pd
import paramiko
from jaxtyping import Array, Float

from . import rkhs, sine

PI_HOST = "172.30.207.3"
PI_USER = "pi"
PI_PASSWORD = os.environ.get("PI_PASSWORD", "R0boT21!")

_sftp: paramiko.SFTPClient | None = None


class LocalStorage:

    def listdir(self, path: str) -> list[str]:
        return os.listdir(path)

    def mkdir(self, path: str) -> None:
        os.mkdir(path)

    def open(self, path: str, mode: str = "r"):
        return open(path, mode.rstrip("b") + "b")


def use_local_storage() -> None:
    global _sftp
    _sftp = LocalStorage()


def _pi():
    global _sftp
    if _sftp is None:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(PI_HOST, username=PI_USER, password=PI_PASSWORD)
        _sftp = client.open_sftp()
    return _sftp


def _makedirs(exp_dir: str) -> None:
    sftp = _pi()
    parents = []
    p = exp_dir.rstrip("/")
    while p and p not in ("/", "."):
        parents.append(p)
        p = posixpath.dirname(p)
    for d in reversed(parents):
        try:
            sftp.mkdir(d)
        except OSError:
            pass


def _indices(exp_dir: str, prefix: str) -> dict[int, str]:
    try:
        names = _pi().listdir(exp_dir)
    except OSError:
        return {}
    found = {}
    for name in names:
        m = re.match(rf"{prefix}_(\d+)\.(txt|json)$", name)
        if m:
            found[int(m.group(1))] = m.group(2)
    return found


def subdirectories(root: str) -> list[tuple[str, str]]:
    sftp = _pi()
    try:
        names = sftp.listdir(root)
    except OSError:
        raise SystemExit(f"No such folder: {root}")
    found = []
    for name in sorted(names):
        path = posixpath.join(root, name)
        try:
            sftp.listdir(path)
        except OSError:
            continue
        found.append((name, path))
    return found


def config_numbers(exp_dir: str) -> set[int]:
    return set(_indices(exp_dir, "config"))


def run_numbers(exp_dir: str) -> set[int]:
    return set(_indices(exp_dir, "run"))


def _config_payload(exp_dir: str, i: int) -> dict:
    with _pi().open(posixpath.join(exp_dir, f"config_{i}.json")) as fh:
        return json.loads(fh.read().decode())


def experiment_mode(exp_dir: str) -> str | None:
    indices = _indices(exp_dir, "config")
    if not indices:
        return None
    legacy = sorted(i for i, ext in indices.items() if ext != "json")
    if legacy:
        raise ValueError(
            f"{exp_dir} holds .txt configs ({legacy[:3]}...), the format used "
            "before both modes became torque profiles. Configs are JSON now; "
            "start a fresh experiment folder."
        )

    mode = _config_payload(exp_dir, min(indices)).get("mode")
    if mode not in ("vector", "functional"):
        raise ValueError(
            f"{exp_dir}: config_{min(indices)}.json has mode {mode!r}, "
            "expected 'vector' or 'functional'."
        )
    return mode


def next_index(exp_dir: str) -> int:
    used = config_numbers(exp_dir) | run_numbers(exp_dir)
    return max(used, default=0) + 1


def read_config(exp_dir: str, i: int) -> Float[Array, "2"]:
    payload = _config_payload(exp_dir, i)
    if payload.get("mode") != "vector":
        raise ValueError(
            f"{exp_dir}/config_{i}.json is a {payload.get('mode')!r} config; "
            "use read_config_function."
        )
    return jnp.array([float(payload[name]) for name in sine.PARAM_NAMES])


def save_config(exp_dir: str, i: int, x: Float[Array, "2"]) -> None:
    if len(x) != sine.DIM:
        raise ValueError(f"vector configs are {sine.PARAM_NAMES}, got {len(x)} values")
    payload = dict(mode="vector", **{n: float(v) for n, v in zip(sine.PARAM_NAMES, x)})
    _makedirs(exp_dir)
    with _pi().open(posixpath.join(exp_dir, f"config_{i}.json"), "w") as fh:
        fh.write(json.dumps(payload, indent=2).encode())


def read_config_profile(exp_dir: str, i: int) -> sine.Sine | rkhs.Function:
    if _config_payload(exp_dir, i).get("mode") == "vector":
        return sine.Sine(read_config(exp_dir, i))
    return read_config_function(exp_dir, i)


PROFILE_RESOLUTION = 64


def profile_grid(resolution: int = PROFILE_RESOLUTION) -> Float[Array, "m 1"]:
    return jnp.linspace(0.0, 1.0, resolution)[:, None]


def save_config_function(exp_dir: str, i: int, f: rkhs.Function) -> None:
    payload = dict(
        mode="functional",
        rho=[float(v) for v in f.rho],
        x=[[float(v) for v in point] for point in f.x],
        a=[float(v) for v in f.a],
    )
    _makedirs(exp_dir)
    with _pi().open(posixpath.join(exp_dir, f"config_{i}.json"), "w") as fh:
        fh.write(json.dumps(payload, indent=2).encode())


def read_config_function(exp_dir: str, i: int) -> rkhs.Function:
    payload = _config_payload(exp_dir, i)
    if payload.get("mode") != "functional":
        raise ValueError(
            f"{exp_dir}/config_{i}.json is a {payload.get('mode')!r} config; "
            "use read_config for its parameters, or read_config_profile for "
            "the curve it commands."
        )
    return rkhs.Function(
        rho=jnp.array(payload["rho"]),
        x=jnp.array(payload["x"]),
        a=jnp.array(payload["a"]),
    )


def load_functional_dataset(
    exp_dir: str,
) -> tuple[list[rkhs.Function], Float[Array, "n"]]:
    completed = sorted(config_numbers(exp_dir) & run_numbers(exp_dir))
    fs = [read_config_function(exp_dir, i) for i in completed]
    ys = [read_result(exp_dir, i) for i in completed]

    dims = {f.x.shape[-1] for f in fs}
    if len(dims) > 1:
        raise ValueError(f"{exp_dir} holds profiles of differing input dims: {dims}")

    rho_min = min((float(f.rho.min()) for f in fs), default=rkhs.RHO_RANGE[0])
    if rho_min < rkhs.RHO_RANGE[0] - 1e-12:
        raise ValueError(
            f"{exp_dir} holds a profile with rho={rho_min:.4g}, below the ambient "
            f"lengthscale {rkhs.RHO_RANGE[0]}; distances would not be finite."
        )
    return fs, jnp.array(ys)


def loss_function(df: pd.DataFrame, beta: float = 5.0) -> float:
    mechanical_power = df[["mechanicalPower_0", "mechanicalPower_1"]].values
    positive = mechanical_power[mechanical_power > 0]
    negative = mechanical_power[mechanical_power < 0]
    return float((positive.sum() + beta * negative.sum()) / mechanical_power.size)


def save_run(exp_dir: str, i: int, df: pd.DataFrame) -> None:
    _makedirs(exp_dir)
    with _pi().open(posixpath.join(exp_dir, f"run_{i}.txt"), "w") as f:
        f.write(df.to_csv(sep=" ", index=False).encode())


def read_result(exp_dir: str, i: int) -> float:
    with _pi().open(posixpath.join(exp_dir, f"run_{i}.txt")) as f:
        text = f.read().decode()
    df = pd.read_csv(io.StringIO(text), sep=" ")
    return loss_function(df)


def load_dataset(exp_dir: str) -> tuple[Float[Array, "n 2"], Float[Array, "n"]]:
    completed = sorted(config_numbers(exp_dir) & run_numbers(exp_dir))
    xs = [read_config(exp_dir, i) for i in completed]
    ys = [read_result(exp_dir, i) for i in completed]
    return jnp.array(xs), jnp.array(ys)


def load_profile_dataset(
    exp_dir: str,
) -> tuple[list[sine.Sine | rkhs.Function], Float[Array, "n"]]:
    completed = sorted(config_numbers(exp_dir) & run_numbers(exp_dir))
    fs = [read_config_profile(exp_dir, i) for i in completed]
    ys = [read_result(exp_dir, i) for i in completed]
    return fs, jnp.array(ys)
