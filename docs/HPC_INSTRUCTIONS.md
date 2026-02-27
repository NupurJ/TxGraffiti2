# Running TxGraffiti on an HPC Cluster

## Overview

Each `graffiti3` run (one call to `Graffiti3.conjecture(...)`) should be a **separate OS process** — one SLURM array job, or one entry in a `ProcessPoolExecutor` pool. Do **not** try to parallelise *within* a single run by spawning sub-processes per stage.

## Recommended approach

```python
from concurrent.futures import ProcessPoolExecutor

def run_one(args):
    g = Graffiti3(...)
    return g.conjecture(target=args.target, ...)

with ProcessPoolExecutor(max_workers=N) as pool:
    results = list(pool.map(run_one, arg_list))
```

Or as a SLURM array job where each task index selects a different target.

## Why not `multiprocessing.Process` per stage?

The obvious alternative — spawn a child process per stage so the parent can `terminate()` it on timeout — is wrong for HPC for several reasons:

1. **Process tree explosion.** N parallel jobs × num_stages processes = potentially hundreds of processes per node. SLURM/PBS enforce per-user process count limits that are easy to blow past.

2. **BLAS/LAPACK fork-safety deadlocks.** numpy and scipy link against OpenBLAS or Intel MKL, which maintain internal thread pools. Forking while those pools are active (the default on Linux with `fork()`) causes the child to inherit a partially-initialised pool, leading to deadlocks. This is a well-known issue.

3. **`Manager().list()` overhead.** The collector pattern that accumulates partial conjectures across the timeout boundary uses a plain Python list. Replacing it with a `Manager().list()` for IPC would add a daemon manager process per job and significant serialisation overhead.

4. **`ThreadPoolExecutor` is also wrong.** `signal.SIGALRM` is only delivered to the main thread. Wrapping graffiti3 calls in threads would silently break the per-stage timeout.

## How the current timeout actually works

- **SIGALRM** (`signal.setitimer(ITIMER_REAL, stage_timeout)`) raises `_StageTimeout` between Python bytecodes. This interrupts pure-Python loops promptly.
- **Solver-level time limits** (`options={"time_limit": solver_time_limit}` passed to every `scipy.optimize.linprog` call) bound individual C-extension LP solves, so Python regains control within roughly `solver_time_limit` seconds of the alarm firing. Without this, a long LP solve would block until it finished, potentially long after the SIGALRM.
- On `_StageTimeout`, partial conjectures accumulated in `_collector` are processed and written out exactly as a normal stage result would be.
- No child processes are created; there are no ghost processes to worry about.

## Files modified to add solver-level time limits

All LP-calling functions now accept `solver_time_limit: Optional[float] = None` and forward it to `linprog`:

| File | Functions |
|------|-----------|
| `runners/lp.py` | `lp_single_runner`, `lp_runner`, `solve_lp_min_slack` |
| `runners/poly.py` | `_solve_lp_upper`, `_solve_lp_lower`, `poly_single_runner` |
| `runners/root_log.py` | `sqrt_single_runner`, `log_single_runner`, `quad_sqrt_runner`, `quad_log_runner` |
| `runners/nonlinear_multi.py` | `_fit_and_build_bounds`, `x_sqrt_log_single_runner`, `sqrt_pair_runner`, `geom_mean_runner`, `sqrt_sum_runner`, `log_sum_runner` |
| `runners/exp_exponent.py` | `exp_exponent_runner` |
| `graffiti3/graffiti3.py` | `_run_stage_runner` (propagates `stage_timeout` → `solver_time_limit`) |

## Setting timeouts

Pass `stage_timeout=<seconds>` to `Graffiti3.conjecture(...)`. The same value is forwarded as `solver_time_limit` to every LP call in that stage, so the wall-clock overrun is bounded to roughly one LP solve duration beyond the alarm.

Reasonable defaults depend on graph size and number of variables. Start with `stage_timeout=60` for interactive use; for overnight HPC runs you may want several hundred seconds per stage.
