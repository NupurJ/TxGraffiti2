# Graffiti3 — Summary of Recent Changes

Changes are listed oldest-first. Each section corresponds to one or more commits.

---

## 1. Verbose logging (`17f9e97`, `d636647`)

**Files:** `graffiti3/graffiti3.py`

- Added a `verbose: bool = False` parameter to `Graffiti3.conjecture(...)`.
- When `verbose=True`, a Python `logging` logger emits each conjecture as it is found (live, during conjecturing) rather than only at the end.
- A second commit added logging of the full conjecture list after each stage completes.

---

## 2. Checkpoint file with summary (`d792209`)

**Files:** `graffiti3/graffiti3.py`

- Checkpoint CSV now includes a human-readable summary section at the top, listing metadata such as target, number of conjectures, and timestamp.

---

## 3. Stage refactor (`ca142fc`)

**Files:** `graffiti3/graffiti3.py`

- Large internal refactor of `_conjecture`. The per-stage logic was extracted into a helper (`_run_stage_runner`) so that the dispatch loop is no longer a giant `if/elif` chain.
- No user-visible behaviour change; this was groundwork for timeouts and checkpointing.

---

## 4. LP runner performance rewrite (`ecea8b9`)

**Files:** `graffiti3/runners/lp.py`

- `lp_runner` was previously routed through `solve_lp_min_slack`, which formulated each LP by adding `n` slack variables — producing an `n × (k+1+n)` dense equality-constrained problem. At `n = 2000`, each solve allocated ~32 MB.
- Replaced with the same direct tight-bound formulation already used by `lp_single_runner`:
  - Upper bound: `min cᵀv  s.t.  −Xc·v_w − b ≥ −y`  (n inequality constraints)
  - Lower bound: `max cᵀv  s.t.   Xc·v_w + b ≤  y`  (n inequality constraints)
- LP matrix is now `n × (k+1)`. At `n = 2000, k = 2` this is ~48 KB vs ~32 MB — **three orders of magnitude smaller**.
- Features are column-centred before solving and un-centred after, improving numerical conditioning.
- `solve_lp_func` parameter retained for API compatibility but silently ignored.
- Result: LP2/LP3/LP4 stages finish in seconds rather than hours.

---

## 5. Write checkpoint on keyboard interrupt (`9d0e8af`)

**Files:** `graffiti3/graffiti3.py`

- `conjecture(...)` now catches `KeyboardInterrupt` and writes whatever conjectures have been found so far to the checkpoint CSV before re-raising (or exiting cleanly, depending on configuration).
- Prevents losing work when a long run is cancelled manually.

---

## 6. Stage column in checkpoint CSV (`292a51d`)

**Files:** `graffiti3/graffiti3.py`

- The checkpoint CSV now records which conjecturing stage each conjecture came from (e.g. `LP1`, `poly`, `sqrt`, …).
- Useful for resuming runs and for diagnosing which stages are productive.

---

## 7. Per-stage timeouts (`60cb294`)

**Files:** `graffiti3/graffiti3.py`, `runners/constant.py`, `runners/exp_exponent.py`, `runners/lp.py`, `runners/mixed.py`, `runners/nonlinear_multi.py`, `runners/poly.py`, `runners/ratio.py`, `runners/root_log.py`

- Added `stage_timeout: Optional[float] = None` parameter to `conjecture(...)`.
- Implemented using `signal.SIGALRM` + `signal.setitimer(ITIMER_REAL, stage_timeout)`. A custom `_StageTimeout` exception is raised between Python bytecodes when the timer fires.
- Runners use the `_collector: Optional[List[Conjecture]]` pattern: conjectures are appended to the list incrementally as they are found. On `_StageTimeout`, the partial list is recovered and processed normally (filtered, heuristic-ranked, written to checkpoint/equalities file) — identical to a stage that completes without a timeout.
- `finally` block always cancels the timer and restores the previous signal handler.
- No child processes are created; there are no background ghost processes.

---

## 8. Read conjectures from checkpoint CSV (`33ace45`)

**Files:** `graffiti3/graffiti3.py`

- Added logic to load previously found conjectures from a checkpoint CSV at the start of a new run.
- Allows resuming interrupted runs: stages whose results are already in the checkpoint are skipped.

---

## 9. Write equalities to a separate file (`5a57aae`)

**Files:** `graffiti3/graffiti3.py`

- After conjecturing, equalities (conjectures that are tight on every graph in the dataset) are written to a dedicated output file, separate from the main conjectures file.
- Makes it easier to identify the strongest results without manually filtering.

---

## 10. Solver-level time limits for LP stages (`b76bb8e`)

**Files:** `graffiti3/graffiti3.py`, `runners/lp.py`, `runners/poly.py`, `runners/root_log.py`, `runners/nonlinear_multi.py`, `runners/exp_exponent.py`

- **Problem:** `SIGALRM` cannot interrupt C-extension calls. If a runner is inside a HiGHS/BLAS call when the alarm fires, the signal is queued until the C call returns — potentially much later.
- **Fix:** Added `solver_time_limit: Optional[float] = None` to every LP-calling function. Passed as `options={"time_limit": solver_time_limit}` to every `scipy.optimize.linprog` call. HiGHS respects this limit and returns control to Python promptly.
- `_run_stage_runner` forwards `stage_timeout` as `solver_time_limit` to all runner calls, so no manual wiring is needed at the call site.
- Functions modified:

| File | Functions |
|------|-----------|
| `runners/lp.py` | `lp_single_runner`, `lp_runner`, `solve_lp_min_slack` |
| `runners/poly.py` | `_solve_lp_upper`, `_solve_lp_lower`, `poly_single_runner` |
| `runners/root_log.py` | `sqrt_single_runner`, `log_single_runner`, `quad_sqrt_runner`, `quad_log_runner` |
| `runners/nonlinear_multi.py` | `_fit_and_build_bounds`, `x_sqrt_log_single_runner`, `sqrt_pair_runner`, `geom_mean_runner`, `sqrt_sum_runner`, `log_sum_runner` |
| `runners/exp_exponent.py` | `exp_exponent_runner` |
| `graffiti3/graffiti3.py` | `_run_stage_runner` |

---

## 11. HPC documentation (`fb202f6`)

**Files:** `docs/HPC_INSTRUCTIONS.md`

- Added guidance on running multiple graffiti3 jobs in parallel on an HPC cluster.
- Explains why `multiprocessing.Process` per stage is wrong for HPC (BLAS fork-safety deadlocks, SLURM process count limits) and what to do instead (`ProcessPoolExecutor` / SLURM array jobs).
- See [HPC_INSTRUCTIONS.md](HPC_INSTRUCTIONS.md) for details.

---

## 12. Inner-loop timeout checks for all LP-calling stages

**Files:** `runners/lp.py`, `runners/poly.py`, `runners/root_log.py`, `runners/nonlinear_multi.py`, `runners/exp_exponent.py`, `runners/mixed.py`, `graffiti3/graffiti3.py`

- **Problem:** Stages whose work is a combinatorial product (hypotheses × pairs of invariants, or hypotheses × invariants × invariants) ran far beyond `stage_timeout`. Even though each individual `linprog` call respected `solver_time_limit`, thousands of such calls accumulated to 5–10× the intended timeout (e.g. 1000+ seconds when `stage_timeout=120`).

- **Affected runners (all runners with nested loops that make LP or ratio calls):**

  | Runner | File | Loop structure |
  |--------|------|----------------|
  | `lp_single_runner` | `runners/lp.py` | hypotheses × invariants |
  | `lp_runner` | `runners/lp.py` | hypotheses × combinations(invariants, k) |
  | `poly_single_runner` | `runners/poly.py` | hypotheses × invariants |
  | `sqrt_single_runner` | `runners/root_log.py` | hypotheses × invariants × invariants |
  | `log_single_runner` | `runners/root_log.py` | hypotheses × invariants × invariants |
  | `quad_sqrt_runner` | `runners/root_log.py` | hypotheses × invariants × invariants |
  | `quad_log_runner` | `runners/root_log.py` | hypotheses × invariants × invariants |
  | `x_sqrt_log_single_runner` | `runners/nonlinear_multi.py` | hypotheses × invariants |
  | `sqrt_pair_runner` | `runners/nonlinear_multi.py` | hypotheses × combinations(invariants, 2) |
  | `geom_mean_runner` | `runners/nonlinear_multi.py` | hypotheses × combinations(invariants, 2) |
  | `sqrt_sum_runner` | `runners/nonlinear_multi.py` | hypotheses × combinations(invariants, 2) |
  | `log_sum_runner` | `runners/nonlinear_multi.py` | hypotheses × combinations(invariants, 2) |
  | `exp_exponent_runner` | `runners/exp_exponent.py` | hypotheses × invariants × invariants |
  | `mixed_runner` | `runners/mixed.py` | hypotheses × primaries × secondaries |

- **Fix:** Added `_stage_timeout: Optional[float] = None` to each runner. When set, `time.perf_counter()` is checked at the top of each hypothesis loop and again at the top of each inner loop. When elapsed time exceeds `_stage_timeout`, `_StageTimeout` is raised immediately, handing control back to the existing partial-result recovery path.

- `_run_stage_runner` in `graffiti3.py` passes `_stage_timeout=stage_timeout` alongside the existing `solver_time_limit=stage_timeout` for all affected runners.

- **Not affected:** `constant_runner` and `ratio_runner` do not make LP calls, so they execute pure-Python code that `SIGALRM` can interrupt between bytecodes without delay. No `_stage_timeout` is needed for these.

- The two timeout parameters serve different roles and are both needed:
  - `solver_time_limit` caps each individual `linprog` call (prevents SIGALRM from being stuck inside a C extension).
  - `_stage_timeout` caps the cumulative wall-clock time across all LP calls in the stage (prevents thousands of short LP calls from adding up to far beyond the timeout).
