# exopt

Bayesian optimization of exoskeleton torque profiles. The controller (the Pi in
real trials) streams power samples over ZMQ and swaps torque profiles on the
fly; the BO driver runs locally, collecting a segment of samples per candidate,
fitting the surrogate, and publishing the next profile — one continuous
experiment, many acquisitions.

## Modes

Both modes optimize a periodic torque profile over the gait cycle `t in [0, 1)`.

- **vector** — one sinusoid: `A * sin(2*pi*(t - phase))`, two parameters.
- **functional** — `k` atoms of a truncated Sobolev (Fourier) RKHS:
  `f(t) = sum_i A_i * kappa_H(t - phi_i)` with
  `kappa_H(d) = sum_{m=1..H} m^-2 * cos(2*pi*m*d)`, so `2k` parameters.

The surrogate is one GP for both: squared exponential of the Sobolev RKHS
distance, which is closed-form on Fourier coefficients. A vector trial is an
`H = 1` atom, so `--mode functional` seamlessly continues a folder of sine
trials with no conversion.

## Running

```bash
# on the host: stream forever, applying whatever profile arrives
uv run mock-stream --target Branin

# reach a remote host through an ssh tunnel
ssh -N -L 5555:localhost:5555 -L 5556:localhost:5556 user@host &

# locally: seed, then segment-collect / fit / propose / publish, in a loop
uv run bo-loop --exp-dir <dir> --duration 30 --warmup 10
```

Each segment runs one profile until its mean converges (`--tol`, capped at
`--max-samples`), with the first `--warmup` samples dropped after a swap.
Segments are recorded to `--exp-dir` as `config_i.json` / `run_i.txt` pairs,
so `bo-loop` resumes an interrupted session, and `--mode functional` continues
a vector one. `bo-loop` maximizes by default (`--minimize` to flip) and keeps
proposing until `--duration` minutes of experiment time elapse.

The real controller needs only a non-blocking `poll()` of the profile socket
in its main loop and a `profile_id` tag on each streamed sample (see
`zmq_link.controller_link`).

## Layout

```
src/exopt/
  bo_loop.py            # bo-loop: BO proposal and the online loop
  gaussian_process.py   # distance-based GP surrogate
  rkhs_functions.py     # Profile atoms and RKHS geometry
  acquisition.py        # log-EI, start designs, restart optimization, loss
  zmq_link.py           # ZMQ pub/sub link and payload encoding
  experiment_io.py      # config_i.json / run_i.txt folder records
  mock_controller.py    # mock-stream: vlse stand-in for the controller
PDO_Walking_profile.py  # host-side real controller: renders profiles on the exo
```
