# src/txgraffiti/graffiti3/graffiti3.py

from __future__ import annotations

from enum import Enum
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .class_relations import GraffitiClassRelations
from .exprs import (
    Expr,
    to_expr,
    abs_ as abs_expr,
    min_ as min_expr,
    max_ as max_expr,
    log as log_expr,
)
from .relations import Conjecture, LeanEnv
from .utils import (
    _filter_by_touch,
    _dedup_conjectures,
    _annotate_and_sort_conjectures,
)
from .types import HypothesisInfo, NonComparablePair, Graffiti3Result, SophieCondition
from .sophie import (
    discover_sophie_from_inequalities,
    rank_sophie_conditions_global,
    print_sophie_conditions,
)

# Runners
from txgraffiti.graffiti3.runners.constant import constant_runner
from txgraffiti.graffiti3.runners.ratio import ratio_runner
from txgraffiti.graffiti3.runners.lp import lp_single_runner, lp_runner
from txgraffiti.graffiti3.runners.mixed import mixed_runner
from txgraffiti.graffiti3.runners.poly import poly_single_runner
from txgraffiti.graffiti3.runners.heuristic import heuristic_runner
from txgraffiti.graffiti3.runners.root_log import sqrt_single_runner, log_single_runner, quad_sqrt_runner, quad_log_runner
from txgraffiti.graffiti3.runners.nonlinear_multi import (
    x_sqrt_log_single_runner,
    sqrt_pair_runner,
    geom_mean_runner,
    sqrt_sum_runner,
    log_sum_runner,
)
from txgraffiti.graffiti3.runners.exp_exponent import exp_exponent_runner

from txgraffiti.graffiti3.relations import sophie_condition_to_lean_theorem
# ───────────────────────── enums ───────────────────────── #

class Stage(str, Enum):
    """Named stages in the Graffiti3 conjecturing pipeline."""
    CONSTANT = "constant"
    RATIO = "ratio"
    LP1 = "lp1"              # 1-feature LP (t ≶ m x + b)
    LP2 = "lp2"                # 2-feature LP
    LP3 = "lp3"              # 3-feature LP
    LP4 = "lp4"
    POLY_SINGLE = "poly_single"
    MIXED = "mixed"
    SQRT = "sqrt"
    LOG = "log"
    SQRT_LOG = "sqrt_log"
    SQRT_PAIR = "sqrt_pair"
    GEOM_MEAN = "geom_mean"
    SQRT_SUM = "sqrt_sum"
    LOG_SUM = "log_sum"
    EXP_EXPONENT = "exp_exponent"


StageLike = Union[Stage, str]


class Mode(str, Enum):
    """
    High level search presets.

    - FAST:    quick, coarse pass with simple features.
    - STANDARD: balanced default (good for most use).
    - DEEP:    more stages & derived features; heavier but richer.
    """
    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"


ModeLike = Union[Mode, str]


# ─────────────────── checkpoint helper ──────────────────────────── #

def _write_conjecture_checkpoint(
    path: str,
    conjectures: "List[Any]",
    target: str,
    stage: str = "",
    stage_timings: Optional[Sequence[Tuple[str, float]]] = None,
) -> None:
    """
    Atomically overwrite *path* with the current conjecture pool.

    Format
    ------
    Top block   : comment lines (# ...) with metadata and stage timings.
    Middle block: CSV rows with columns: n,conjecture,touches,support
    Bottom block: summary line appended later by _append_checkpoint_summary.

    Uses a .tmp sibling file + rename so readers never see a partial write.
    """
    import csv
    import io
    import os
    import tempfile
    from datetime import datetime

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stage_tag = f" | after stage '{stage}'" if stage else ""

    header_lines: List[str] = []
    header_lines.append(f"# Checkpoint: target='{target}'{stage_tag}")
    header_lines.append(f"# Written: {ts}")
    header_lines.append(f"# Conjectures: {len(conjectures)}")
    if stage_timings:
        header_lines.append("# Stages run (order, seconds):")
        for i, (name, seconds) in enumerate(stage_timings, 1):
            header_lines.append(f"#   {i}. {name}: {seconds:.2f}s")
    header_lines.append("")

    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf, quoting=csv.QUOTE_ALL)
    writer.writerow(["n", "conjecture", "touches", "support"])
    for i, c in enumerate(conjectures, 1):
        try:
            text = c.pretty()
        except Exception:
            text = repr(c)
        touches = getattr(c, "touch_count", getattr(c, "touch", ""))
        support = getattr(c, "support_n", getattr(c, "support", ""))
        writer.writerow([i, text, touches, support])

    content = "\n".join(header_lines) + csv_buf.getvalue()

    dir_ = os.path.dirname(os.path.abspath(path)) or "."
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_path, path)
    except Exception:
        # Never crash the main pipeline over a checkpoint write
        pass


def _append_checkpoint_summary(path: str, summary: str) -> None:
    """Append a summary block to the end of an existing checkpoint file."""
    if not summary:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n")
            fh.write("# Summary\n")
            fh.write(summary.strip() + "\n")
    except Exception:
        # Never crash the main pipeline over a checkpoint write
        pass


# ───────────────────────── main workspace ───────────────────────── #

class Graffiti3:
    """
    Refactored, modular conjecturing workspace.

    Typical usage
    -------------
        g3 = Graffiti3(df)

        # simplest: one target, standard search
        result = g3.conjecture("independence_number")

        # fast / deep presets
        result_fast  = g3.conjecture("independence_number", mode=Mode.FAST)
        result_deep  = g3.conjecture("independence_number", mode=Mode.DEEP)

        # batch over several targets
        results = g3.list_conjecture(["independence_number", "domination_number"])

    What Graffiti3 precomputes
    --------------------------
    - base_property, base_name, base_mask
    - invariants: Exprs for numeric columns
    - invariant_products: pairwise products of invariants
    - log_exprs: log(x) for safely >1 numeric invariants
    - properties: atomic boolean columns
    - hypotheses: nonredundant conjunctions of properties with base
    - non_comparable_invariants: (x, y) where both x<y and x>y occur
    - abs_exprs, min_max_exprs on non-comparable pairs
    """

    # ──────────────────────── construction ────────────────────────

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        max_boolean_arity: int = 2,
        morgan_filter=None,
        dalmatian_filter=None,
        sophie_cfg: Optional[Dict[str, Any]] = None,
        min_touches: int = 3,
        lean_label: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        df : DataFrame
            Invariant / property table (rows = objects, columns = invariants + booleans).
        max_boolean_arity : int, default=2
            Max size of conjunctions used to form hypotheses.
        morgan_filter, dalmatian_filter
            Optional callable heuristics used by `heuristic_runner`.
        sophie_cfg : dict, optional
            Optional overrides for Sophie mining config (eq_tol, min_target_support, etc.).
        min_touches : int, default=3
            Minimum #equalities required for a conjecture to enter heuristics/Sophie.
        """
        self.df = df.copy()

        # Optional Lean4 labels (column/property -> Lean term/type)
        self.lean_env: LeanEnv = LeanEnv.from_mapping(lean_label)
        self.lean_label = lean_label
        self.gcr = GraffitiClassRelations(self.df)

        # heuristic hooks
        self.morgan_filter = morgan_filter
        self.dalmatian_filter = dalmatian_filter

        # Sophie configuration: sensible defaults + user overrides
        cfg = self._default_sophie_cfg()
        if sophie_cfg:
            cfg.update(sophie_cfg)
        self.sophie_cfg: Dict[str, Any] = cfg

        # low-touch pruning threshold
        self.min_touches = int(min_touches)

        # base universe
        self.base_property = self.gcr.base_hypothesis
        self.base_name = self.gcr.base_hypothesis_name
        self.base_mask = np.asarray(
            self.gcr._mask_cached(self.gcr.base_hypothesis), dtype=bool
        )

        # numeric invariants as Exprs
        self.invariants: Dict[str, Expr] = self._build_invariants()

        # boolean properties
        self.properties: List[str] = list(self.gcr.boolean_cols)

        # hypotheses (arity ≤ max_boolean_arity), base first
        self.hypotheses: List[HypothesisInfo] = self._build_hypotheses(
            max_boolean_arity=max_boolean_arity
        )

        # derived numeric Expr pools
        self.invariant_products: Dict[str, Expr] = self._build_invariant_products()
        self.non_comparable_pairs: List[NonComparablePair] = (
            self._find_non_comparable_pairs()
        )
        self.abs_exprs: Dict[str, Expr] = self._build_abs_exprs()
        self.min_max_exprs: Dict[str, Expr] = self._build_min_max_exprs()
        self.log_exprs: Dict[str, Expr] = self._build_log_exprs(
            min_value=1.0,
            base=None,  # natural log by default
        )

    # ──────────────────────── defaults / config ────────────────────────

    @staticmethod
    def _default_sophie_cfg() -> Dict[str, Any]:
        """Baseline settings for Sophie-style condition discovery."""
        return dict(
            eq_tol=1e-4,
            min_target_support=5,
            min_h_support=3,
            max_violations=0,
            min_new_coverage=1,
        )

    # ──────────────────────── internal builders ────────────────────────

    def _build_invariants(self) -> Dict[str, Expr]:
        """Numeric DataFrame columns → Expr."""
        out: Dict[str, Expr] = {}
        for col in self.df.columns:
            if np.issubdtype(self.df[col].dtype, np.number):
                out[col] = to_expr(col)
        return out

    def _build_hypotheses(self, max_boolean_arity: int) -> List[HypothesisInfo]:
        """
        Build nontrivial hypotheses inside the base universe, avoiding
        trivial or redundant conjunctions.

        1. Ask GraffitiClassRelations for nonredundant conjunctions (arity ≤ max_boolean_arity).
        2. Restrict each mask to the base universe.
        3. Discard empty masks and those equal to the base mask.
        4. For identical masks, keep the simplest label (fewest '&' parts).
        5. Return HypothesisInfo objects, with base first.
        """
        self.gcr.enumerate_conjunctions(max_arity=max_boolean_arity)
        nonred, _, _ = self.gcr.find_redundant_conjunctions()

        hyps: List[HypothesisInfo] = []

        base_info = HypothesisInfo(
            name=self.base_name,
            pred=self.base_property,
            mask=self.base_mask.copy(),
        )
        hyps.append(base_info)

        def _complexity(label: str) -> int:
            parts = [p.strip() for p in label.split("&") if p.strip()]
            return max(1, len(parts))

        seen: Dict[bytes, Tuple[int, HypothesisInfo]] = {}
        base_key = self.base_mask.tobytes()
        seen[base_key] = (0, base_info)

        for name, pred in nonred:
            raw_mask = np.asarray(self.gcr._mask_cached(pred), dtype=bool)
            mask = raw_mask & self.base_mask
            if not mask.any():
                continue

            key = mask.tobytes()
            comp = _complexity(name)

            if key in seen:
                old_comp, _old_info = seen[key]
                if comp >= old_comp:
                    continue
            seen[key] = (comp, HypothesisInfo(name=name, pred=pred, mask=mask))

        for _key, (_comp, info) in seen.items():
            if info.name == self.base_name:
                continue
            hyps.append(info)

        return hyps

    def _build_invariant_products(self) -> Dict[str, Expr]:
        """All pairwise products of numeric invariants."""
        cols = list(self.invariants.keys())
        out: Dict[str, Expr] = {}
        for i in range(len(cols)):
            for j in range(i, len(cols)):
                a, b = cols[i], cols[j]
                name = f"{a}·{b}" if a != b else f"{a}²"
                out[name] = self.invariants[a] * self.invariants[b]
        return out

    def _build_log_exprs(
        self,
        *,
        min_value: float = 1.0,
        base: float | None = None,
    ) -> Dict[str, Expr]:
        """
        Build log invariants log(x) for numeric columns x that are safely > min_value
        on the base universe.
        """
        out: Dict[str, Expr] = {}
        bm = self.base_mask

        for name, _expr in self.invariants.items():
            col = self.df[name].to_numpy(dtype=float)
            vals = col[bm]

            finite = np.isfinite(vals)
            if finite.sum() == 0:
                continue

            safe_vals = vals[finite]
            if np.all(safe_vals > min_value):
                if base is None:
                    log_name = f"ln_{name}"
                else:
                    b = float(base)
                    if b.is_integer():
                        log_name = f"log{int(b)}_{name}"
                    else:
                        log_name = f"log_{b}_{name}"

                out[log_name] = log_expr(name, base=base, epsilon=0.0)

        return out

    def _find_non_comparable_pairs(self) -> List[NonComparablePair]:
        """
        Find pairs (x, y) such that, restricted to the base universe,
        there exist rows with x < y and rows with x > y.
        """
        cols = list(self.invariants.keys())
        vals = {c: self.df[c].to_numpy(dtype=float) for c in cols}
        bm = self.base_mask
        out: List[NonComparablePair] = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                xa = vals[a][bm]
                xb = vals[b][bm]

                finite_mask = np.isfinite(xa) & np.isfinite(xb)
                if finite_mask.sum() == 0:
                    continue

                x = xa[finite_mask]
                y = xb[finite_mask]

                lt = (x < y).any()
                gt = (x > y).any()
                if lt and gt:
                    out.append(
                        NonComparablePair(
                            left=self.invariants[a],
                            right=self.invariants[b],
                            left_name=a,
                            right_name=b,
                        )
                    )
        return out

    def _build_abs_exprs(self) -> Dict[str, Expr]:
        """abs(x - y) for each non-comparable pair (x, y)."""
        out: Dict[str, Expr] = {}
        for pair in self.non_comparable_pairs:
            name = f"|{pair.left_name} - {pair.right_name}|"
            out[name] = abs_expr(pair.left - pair.right)
        return out

    def _build_min_max_exprs(self) -> Dict[str, Expr]:
        """min(x,y) and max(x,y) for each non-comparable pair (x, y)."""
        out: Dict[str, Expr] = {}
        for pair in self.non_comparable_pairs:
            name_min = f"min({pair.left_name}, {pair.right_name})"
            name_max = f"max({pair.left_name}, {pair.right_name})"
            out[name_min] = min_expr(pair.left, pair.right)
            out[name_max] = max_expr(pair.left, pair.right)
        return out

    # ── helper: build others pool from flags ──

    def _build_others_pool(
        self,
        target: str,
        *,
        include_invariant_products: bool,
        include_abs: bool,
        include_min_max: bool,
        include_log: bool,
    ) -> Dict[str, Expr]:
        """
        Construct the dictionary of available "other" expressions, respecting
        the boolean flags and removing anything that obviously depends on
        the target column.
        """
        target_str = target

        def _uses_target(expr: Expr) -> bool:
            return target_str in repr(expr)

        pool: Dict[str, Expr] = {}

        # base numeric invariants
        for name, expr in self.invariants.items():
            if name == target:
                continue
            pool[name] = expr

        if include_invariant_products:
            for name, expr in self.invariant_products.items():
                if _uses_target(expr):
                    continue
                pool[name] = expr

        if include_abs:
            for name, expr in self.abs_exprs.items():
                if _uses_target(expr):
                    continue
                pool[name] = expr

        if include_min_max:
            for name, expr in self.min_max_exprs.items():
                if _uses_target(expr):
                    continue
                pool[name] = expr

        if include_log:
            for name, expr in self.log_exprs.items():
                if _uses_target(expr):
                    continue
                pool[name] = expr

        return pool

    # ── helper: map complexity ↦ default stages ──

    @staticmethod
    def _stages_from_complexity(complexity: int) -> List[Stage]:
        """
        Map an integer complexity level to a default list of stages.

        0 → [CONSTANT]
        1 → + [RATIO, LP1]
        2 → + [LP, POLY_SINGLE, SQRT, LOG]
        3 → + [LP3, SQRT_LOG]
        ≥4 → + [MIXED, SQRT_PAIR, GEOM_MEAN, SQRT_SUM, LOG_SUM]
        """
        stages: List[Stage] = [Stage.CONSTANT]
        if complexity >= 1:
            stages += [Stage.RATIO, Stage.LP1]
        if complexity >= 2:
            stages += [Stage.LP2, Stage.POLY_SINGLE, Stage.SQRT, Stage.LOG]
        if complexity >= 3:
            stages += [Stage.LP3, Stage.SQRT_LOG]
        if complexity >= 4:
            stages += [
                Stage.MIXED,
                Stage.SQRT_PAIR,
                Stage.GEOM_MEAN,
                Stage.SQRT_SUM,
                Stage.LOG_SUM,
            ]
        return stages

    @staticmethod
    def _normalize_stage_list(
        raw: Sequence[StageLike],
        *,
        param_name: str,
    ) -> List[Stage]:
        """
        Normalize a sequence of Stage / string into a deduplicated list of Stage.
        Raises on unknown strings.
        """
        out: List[Stage] = []
        seen: set[Stage] = set()
        for item in raw:
            if isinstance(item, Stage):
                stage = item
            else:
                try:
                    stage = Stage(item)
                except ValueError as e:
                    allowed = ", ".join(s.value for s in Stage)
                    raise ValueError(
                        f"Unknown stage in `{param_name}`: {item!r}. "
                        f"Allowed: {allowed}"
                    ) from e
            if stage not in seen:
                seen.add(stage)
                out.append(stage)
        return out

    @staticmethod
    def _normalize_mode(mode: ModeLike) -> Mode:
        """Normalize Mode / string to a Mode enum."""
        if isinstance(mode, Mode):
            return mode
        try:
            return Mode(mode)
        except ValueError as e:
            allowed = ", ".join(m.value for m in Mode)
            raise ValueError(
                f"Unknown mode {mode!r}. Allowed modes: {allowed}"
            ) from e

    @staticmethod
    def _presets_for_mode(mode: Mode) -> Dict[str, Any]:
        """
        Map Mode → presets for complexity / feature flags / quick.
        """
        if mode is Mode.FAST:
            return dict(
                complexity=1,
                include_invariant_products=False,
                include_abs=False,
                include_min_max=False,
                include_log=False,
                quick=True,
            )
        if mode is Mode.STANDARD:
            return dict(
                complexity=2,
                include_invariant_products=False,
                include_abs=True,
                include_min_max=True,
                include_log=True,
                quick=False,
            )
        if mode is Mode.DEEP:
            return dict(
                complexity=3,
                include_invariant_products=True,
                include_abs=True,
                include_min_max=True,
                include_log=True,
                quick=False,
            )
        raise AssertionError("unreachable")

    def _select_relevant_others(
        self,
        target: str,
        others: Dict[str, Expr],
        *,
        min_corr: float = 0.1,
        max_candidates: int | None = 200,
    ) -> Dict[str, Expr]:
        """
        Cheap pre-filter: keep only those 'other' expressions that show
        at least `min_corr` absolute Pearson correlation with the target
        over the base universe, up to `max_candidates` features.
        """
        y_full = pd.to_numeric(self.df[target], errors="coerce").astype(float)
        y = y_full[self.base_mask]

        scores: List[Tuple[float, str, Expr]] = []

        for name, expr in others.items():
            vals = expr.eval(self.df)[self.base_mask]

            m = y.notna() & vals.notna()
            if m.sum() < 5:
                continue

            yv = y[m].to_numpy(dtype=float)
            gv = vals[m].to_numpy(dtype=float)

            if np.allclose(gv, gv[0]):
                continue

            try:
                c = np.corrcoef(yv, gv)[0, 1]
            except Exception:
                continue
            if not np.isfinite(c):
                continue

            score = float(abs(c))
            if score >= min_corr:
                scores.append((score, name, expr))

        scores.sort(key=lambda t: t[0], reverse=True)

        filtered: Dict[str, Expr] = {}
        for _score, name, expr in scores:
            filtered[name] = expr
            if max_candidates is not None and len(filtered) >= max_candidates:
                break

        return filtered

    # ──────────────────────── public API ────────────────────────

    def _conjecture(
        self,
        target: str,
        complexity: Optional[int] = None,
        *,
        mode: Optional[ModeLike] = None,
        stages: Optional[Sequence[StageLike]] = None,
        include_invariant_products: Optional[bool] = None,
        include_abs: Optional[bool] = None,
        include_min_max: Optional[bool] = None,
        include_log: Optional[bool] = None,
        enable_sophie: bool = True,
        sophie_stages: Optional[Sequence[StageLike]] = None,
        quick: Optional[bool] = None,
        verbose: bool = True,
        checkpoint_file: Optional[str] = None,
    ) -> Graffiti3Result:
        """
        Main conjecturing driver for a single target.

        Parameters
        ----------
        target : str
            Target invariant column.
        complexity : int, optional
            Coarse knob for enabling groups of stages; ignored if `stages` is provided.
            If None, defaults to 2 (constant + ratio + LP1 + LP + poly_single)
            unless overridden by `mode`.
        mode : Mode or {"fast","standard","deep"}, optional
            High-level preset. If provided, it sets *defaults* for complexity and
            feature flags; you can override them explicitly by passing the
            corresponding arguments.
        stages : sequence of Stage or str, optional
            Explicit list of stages to run. If given, overrides `complexity`.
        include_invariant_products / include_abs / include_min_max / include_log : bool, optional
            Feature toggles. If None, they are inferred from `mode` (if given) or
            fall back to sensible defaults.
        enable_sophie : bool, default=True
            If False, skip all Sophie mining and return an empty list.
        sophie_stages : sequence of Stage or str, optional
            If provided, Sophie is mined only from these stages (subset of `stages`).
            If None, Sophie is mined from all active stages.
        quick : bool, optional
            If True, run a cheap correlation-based prefilter over the `others` pool
            before expensive LP/poly/mixed stages. If None, inferred from `mode`.

        Returns
        -------
        Graffiti3Result
        """
        if target not in self.df.columns:
            raise KeyError(f"Target column '{target}' not found.")

        # ── 1. Apply mode presets to fill in missing knobs ──────────────
        if mode is not None:
            mode_enum = self._normalize_mode(mode)
            presets = self._presets_for_mode(mode_enum)

            if complexity is None:
                complexity = presets["complexity"]
            if include_invariant_products is None:
                include_invariant_products = presets["include_invariant_products"]
            if include_abs is None:
                include_abs = presets["include_abs"]
            if include_min_max is None:
                include_min_max = presets["include_min_max"]
            if include_log is None:
                include_log = presets["include_log"]
            if quick is None:
                quick = presets["quick"]

        # Fallback defaults when no mode / partial overrides
        if complexity is None:
            complexity = 2
        if include_invariant_products is None:
            include_invariant_products = False
        if include_abs is None:
            include_abs = True
        if include_min_max is None:
            include_min_max = True
        if include_log is None:
            include_log = True
        if quick is None:
            quick = False

        # ── 2. Decide stages ────────────────────────────────────────────
        if stages is not None:
            stages_to_run: List[Stage] = self._normalize_stage_list(
                stages, param_name="stages"
            )
        else:
            stages_to_run = self._stages_from_complexity(int(complexity))

        stages_to_run_set = set(stages_to_run)

        # Sophie stage selection
        if sophie_stages is not None:
            sophie_stage_list = self._normalize_stage_list(
                sophie_stages, param_name="sophie_stages"
            )
            sophie_stage_set = set(sophie_stage_list)
        else:
            sophie_stage_set = stages_to_run_set

        target_expr = to_expr(target)
        min_touches = self.min_touches
        _t0 = time.perf_counter()

        if checkpoint_file:
            import os
            from datetime import datetime as _dt
            _cp_stem, _cp_ext = os.path.splitext(checkpoint_file)
            _cp_ext = _cp_ext or ".txt"
            checkpoint_file = f"{_cp_stem}_{_dt.now().strftime('%Y%m%d_%H%M%S')}{_cp_ext}"

        if verbose:
            print(
                f"[Graffiti3] target='{target}' | "
                f"{len(stages_to_run)} stage(s): {[s.value for s in stages_to_run]}",
                flush=True,
            )
            print(
                f"[Graffiti3] dataset: {len(self.df)} rows, "
                f"{len(self.df.columns)} columns: {list(self.df.columns)}",
                flush=True,
            )

        # ── 3. Build others pool (and maybe prefilter) ─────────────────
        others: Dict[str, Expr] = self._build_others_pool(
            target=target,
            include_invariant_products=bool(include_invariant_products),
            include_abs=bool(include_abs),
            include_min_max=bool(include_min_max),
            include_log=bool(include_log),
        )

        if quick and others:
            others = self._select_relevant_others(target=target, others=others)

        all_conjectures: List[Conjecture] = []
        all_sophie: List[SophieCondition] = []
        stage_info: Dict[str, Any] = {}
        stage_timings: List[Tuple[str, float]] = []

        def _maybe_sophie(stage: Stage, conjs: List[Conjecture]) -> List[SophieCondition]:
            if not enable_sophie or not conjs:
                return []
            if stage not in sophie_stage_set:
                return []
            return sophie_runner(conjectures=conjs, g4=self)

        # ── Execute stages in the order specified by stages_to_run ──────
        for sn, cur_stage in enumerate(stages_to_run, 1):
            if verbose:
                print(
                    f"  [{sn}/{len(stages_to_run)}] Running '{cur_stage.value}'..."
                    f" (+{time.perf_counter() - _t0:.1f}s)",
                    flush=True,
                )
            _stage_t0 = time.perf_counter()

            if cur_stage == Stage.CONSTANT:
                stage_conjs = constant_runner(
                    target_col=target,
                    target_expr=target_expr,
                    hypotheses=self.hypotheses,
                    df=self.df,
                )
                stage_conjs = _filter_by_touch(self.df, stage_conjs, min_touches)

            elif cur_stage == Stage.RATIO:
                stage_conjs = ratio_runner(
                    target_col=target,
                    target_expr=target_expr,
                    others=others,
                    hypotheses=self.hypotheses,
                    df=self.df,
                )
                stage_conjs = _filter_by_touch(self.df, stage_conjs, min_touches)

            elif cur_stage == Stage.LP1:
                stage_conjs = lp_single_runner(
                    target_col=target,
                    target_expr=target_expr,
                    others=others,
                    hypotheses=self.hypotheses,
                    df=self.df,
                    direction="both",
                )
                stage_conjs = _filter_by_touch(self.df, stage_conjs, min_touches)

            elif cur_stage == Stage.LP2:
                stage_conjs = lp_runner(
                    target_col=target,
                    target_expr=target_expr,
                    others=others,
                    hypotheses=self.hypotheses,
                    df=self.df,
                    max_features=2,
                    max_denom=20,
                    coef_bound=10.0,
                    direction="both",
                )
                stage_conjs = _filter_by_touch(self.df, stage_conjs, min_touches)

            elif cur_stage == Stage.LP3:
                stage_conjs = lp_runner(
                    target_col=target,
                    target_expr=target_expr,
                    others=others,
                    hypotheses=self.hypotheses,
                    df=self.df,
                    max_features=3,
                    max_denom=20,
                    coef_bound=10.0,
                    direction="both",
                )
                stage_conjs = _filter_by_touch(self.df, stage_conjs, min_touches)

            elif cur_stage == Stage.LP4:
                stage_conjs = lp_runner(
                    target_col=target,
                    target_expr=target_expr,
                    others=others,
                    hypotheses=self.hypotheses,
                    df=self.df,
                    max_features=4,
                    max_denom=30,
                    coef_bound=10.0,
                    direction="both",
                )
                stage_conjs = _filter_by_touch(self.df, stage_conjs, min_touches)

            elif cur_stage == Stage.POLY_SINGLE:
                stage_conjs = poly_single_runner(
                    target_col=target,
                    target_expr=target_expr,
                    others=others,
                    hypotheses=self.hypotheses,
                    df=self.df,
                    min_support=8,
                    max_denom=20,
                    max_coef_abs=4.0,
                )
                # poly_single_runner already filters internally; no touch filter here

            elif cur_stage == Stage.MIXED:
                stage_conjs = mixed_runner(
                    target_col=target,
                    target_expr=target_expr,
                    primaries=self.invariants,
                    secondaries=self.invariants,
                    hypotheses=self.hypotheses,
                    df=self.df,
                    weight=0.5,
                )
                # mixed_runner does not guarantee min_touches; skip extra filter

            elif cur_stage == Stage.SQRT:
                _sqrt_single = sqrt_single_runner(
                    target_col=target,
                    target_expr=target_expr,
                    others=others,
                    hypotheses=self.hypotheses,
                    df=self.df,
                    max_denom=30,
                    coef_bound=10.0,
                )
                _sqrt_quad = quad_sqrt_runner(
                    target_col=target,
                    target_expr=target_expr,
                    others=others,
                    hypotheses=self.hypotheses,
                    df=self.df,
                    max_denom=30,
                    coef_bound=10.0,
                )
                stage_conjs = _filter_by_touch(self.df, _sqrt_single + _sqrt_quad, min_touches)

            elif cur_stage == Stage.LOG:
                _log_single = log_single_runner(
                    target_col=target,
                    target_expr=target_expr,
                    others=others,
                    hypotheses=self.hypotheses,
                    df=self.df,
                    max_denom=30,
                    coef_bound=10.0,
                    log_epsilon=1e-6,
                    log_base=2,
                )
                _log_quad = quad_log_runner(
                    target_col=target,
                    target_expr=target_expr,
                    others=others,
                    hypotheses=self.hypotheses,
                    df=self.df,
                    max_denom=30,
                    coef_bound=10.0,
                    log_epsilon=1e-6,
                    log_base=2,
                )
                stage_conjs = _filter_by_touch(self.df, _log_single + _log_quad, min_touches)

            elif cur_stage == Stage.SQRT_LOG:
                stage_conjs = x_sqrt_log_single_runner(
                    target_col=target,
                    target_expr=target_expr,
                    others=others,
                    hypotheses=self.hypotheses,
                    df=self.df,
                    min_support=8,
                    max_denom=30,
                    coef_bound=4.0,
                    zero_tol=1e-8,
                    max_coef_abs=4.0,
                    max_intercept_abs=8.0,
                    log_base=None,
                    log_epsilon=1e-6,
                )
                stage_conjs = _filter_by_touch(self.df, stage_conjs, min_touches)

            elif cur_stage == Stage.SQRT_PAIR:
                stage_conjs = sqrt_pair_runner(
                    target_col=target,
                    target_expr=target_expr,
                    others=others,
                    hypotheses=self.hypotheses,
                    df=self.df,
                    min_support=8,
                    max_denom=30,
                    coef_bound=4.0,
                    zero_tol=1e-8,
                    max_coef_abs=4.0,
                    max_intercept_abs=8.0,
                )
                stage_conjs = _filter_by_touch(self.df, stage_conjs, min_touches)

            elif cur_stage == Stage.GEOM_MEAN:
                stage_conjs = geom_mean_runner(
                    target_col=target,
                    target_expr=target_expr,
                    others=others,
                    hypotheses=self.hypotheses,
                    df=self.df,
                    min_support=8,
                    max_denom=30,
                    coef_bound=4.0,
                    zero_tol=1e-8,
                    max_coef_abs=4.0,
                    max_intercept_abs=8.0,
                )
                stage_conjs = _filter_by_touch(self.df, stage_conjs, min_touches)

            elif cur_stage == Stage.SQRT_SUM:
                stage_conjs = sqrt_sum_runner(
                    target_col=target,
                    target_expr=target_expr,
                    others=others,
                    hypotheses=self.hypotheses,
                    df=self.df,
                    min_support=8,
                    max_denom=30,
                    coef_bound=4.0,
                    zero_tol=1e-8,
                    max_coef_abs=4.0,
                    max_intercept_abs=8.0,
                )
                stage_conjs = _filter_by_touch(self.df, stage_conjs, min_touches)

            elif cur_stage == Stage.LOG_SUM:
                stage_conjs = log_sum_runner(
                    target_col=target,
                    target_expr=target_expr,
                    others=others,
                    hypotheses=self.hypotheses,
                    df=self.df,
                    min_support=8,
                    max_denom=30,
                    coef_bound=4.0,
                    zero_tol=1e-8,
                    max_coef_abs=4.0,
                    max_intercept_abs=8.0,
                    log_base=None,
                    log_epsilon=1e-6,
                )
                stage_conjs = _filter_by_touch(self.df, stage_conjs, min_touches)

            elif cur_stage == Stage.EXP_EXPONENT:
                stage_conjs = exp_exponent_runner(
                    target_col=target,
                    target_expr=target_expr,
                    others=others,
                    hypotheses=self.hypotheses,
                    df=self.df,
                    min_support=8,
                    max_denom=30,
                    coef_bound=4.0,
                    zero_tol=1e-8,
                    log_base=None,
                    log_epsilon=1e-6,
                )
                stage_conjs = _filter_by_touch(self.df, stage_conjs, min_touches)

            else:
                # Unknown stage — skip gracefully
                continue

            stage_conjs = heuristic_runner(
                stage_conjs,
                df=self.df,
                morgan_filter=self.morgan_filter,
                dalmatian_filter=self.dalmatian_filter,
            )
            stage_sophie = _maybe_sophie(cur_stage, stage_conjs)

            all_conjectures.extend(stage_conjs)
            all_conjectures = _dedup_conjectures(all_conjectures)
            stage_elapsed = time.perf_counter() - _stage_t0
            stage_timings.append((cur_stage.value, stage_elapsed))
            if checkpoint_file:
                _write_conjecture_checkpoint(
                    checkpoint_file,
                    all_conjectures,
                    target,
                    cur_stage.value,
                    stage_timings=stage_timings,
                )
            all_sophie.extend(stage_sophie)
            stage_info[cur_stage.value] = dict(
                conjectures=len(stage_conjs),
                sophie=len(stage_sophie),
                time_s=stage_elapsed,
            )

        # ── Final pass: annotate & sort ───────────────────────────────
        all_conjectures = _annotate_and_sort_conjectures(self.df, all_conjectures)

        # ── Sophie: deduplicate and globally rank ─────────────────────
        if enable_sophie and all_sophie:
            all_sophie_ranked = rank_sophie_conditions_global(
                _group_sophie_by_property(all_sophie)
            )
        else:
            all_sophie_ranked = []

        summary_line = (
            f"[Graffiti3] Done. "
            f"{len(all_conjectures)} conjecture(s), "
            f"{len(all_sophie_ranked)} sophie condition(s). "
            f"(total: {time.perf_counter() - _t0:.1f}s)"
        )
        if verbose:
            print(summary_line, flush=True)
        if checkpoint_file:
            _append_checkpoint_summary(checkpoint_file, summary_line)

        return Graffiti3Result(
            target=target,
            conjectures=all_conjectures,
            sophie_conditions=all_sophie_ranked,
            stage_breakdown=stage_info,
        )

    def conjecture(
        self,
        targets: Sequence[str],
        *,
        complexity: Optional[int] = None,
        mode: Optional[ModeLike] = None,
        stages: Optional[Sequence[StageLike]] = None,
        include_invariant_products: Optional[bool] = None,
        include_abs: Optional[bool] = None,
        include_min_max: Optional[bool] = None,
        include_log: Optional[bool] = None,
        enable_sophie: bool = True,
        sophie_stages: Optional[Sequence[StageLike]] = None,
        quick: Optional[bool] = None,
        verbose: bool = True,
        show: Optional[bool] = None,
        show_k_conjectures: Optional[int] = 20,
        checkpoint_file: Optional[str] = None,
    ) -> Graffiti3Result:
        """
        Batch version of `conjecture` over multiple targets.

        Returns a single Graffiti3Result aggregating all conjectures and Sophie
        conditions, with `stage_breakdown` containing a per-target summary.
        """
        all_conjs: List[Conjecture] = []
        all_sophie: List[SophieCondition] = []
        breakdown: Dict[str, Any] = {}

        _targets = list(targets)
        for i, t in enumerate(_targets):
            if verbose and len(_targets) > 1:
                print(
                    f"[Graffiti3] ── Target {i+1}/{len(_targets)}: '{t}' ──",
                    flush=True,
                )
            res = self._conjecture(
                t,
                complexity=complexity,
                mode=mode,
                stages=stages,
                include_invariant_products=include_invariant_products,
                include_abs=include_abs,
                include_min_max=include_min_max,
                include_log=include_log,
                enable_sophie=enable_sophie,
                sophie_stages=sophie_stages,
                quick=quick,
                verbose=verbose,
                checkpoint_file=checkpoint_file,
            )
            all_conjs.extend(res.conjectures)
            all_sophie.extend(res.sophie_conditions)
            breakdown[t] = res.stage_breakdown

        # Re-run annotation / sorting globally
        all_conjs = _annotate_and_sort_conjectures(self.df, all_conjs)

        if enable_sophie and all_sophie:
            all_sophie_ranked = rank_sophie_conditions_global(
                _group_sophie_by_property(all_sophie)
            )
        else:
            all_sophie_ranked = []

        # This "target" is just a label in the combined result
        result = Graffiti3Result(
            target=",".join(targets),
            conjectures=all_conjs,
            sophie_conditions=all_sophie_ranked,
            stage_breakdown=breakdown,
        )

        if show:
            print_g3_result(
                result,
                k_conjectures=show_k_conjectures,
                k_sophie=show_k_conjectures,
            )
        self.result = result
        if self.lean_label:
            self.lean_necessary_statements = self.conjectures_as_lean(all_conjs, prefix="Necessary_Condition", start_index=1)
            self.lean_sufficient_statements = self.sophie_conditions_as_lean(all_sophie_ranked, prefix="Sufficient_Condition", start_index=1)
        return result


    # ──────────────────────── Lean4 utilities ────────────────────────

    from typing import Sequence, List, Optional

    def conjectures_as_lean(
        self,
        conjectures: Optional[Sequence["Conjecture"]] = None,
        *,
        prefix: str = "TxGraffitiBench",
        start_index: int = 1,
        include_sorry: bool = True,
        attach: bool = False,
    ) -> List[str]:
        """Render conjectures as Lean4 theorem stubs."""
        if not self.lean_env or not self.lean_env.labels:
            raise ValueError("No Lean labels provided (pass lean_label=... to Graffiti3).")

        if conjectures is None:
            if self.result is None:
                raise ValueError("No conjectures provided and self.result is None.")
            conjectures = self.result.conjectures

        out: List[str] = []
        for k, c in enumerate(conjectures, start=start_index):
            thm = f"{prefix}_{k}"
            s = c.to_lean_theorem(
                self.lean_env,
                thm,
                base_condition=self.base_property,
                include_sorry=include_sorry,
            )
            out.append(s)
            if attach:
                setattr(c, "lean_theorem", s)

        return out


    def sophie_conditions_as_lean(
        self,
        sophie_conditions: Optional[Sequence["SophieCondition"]] = None,
        *,
        prefix: str = "TxGraffitiSophie",
        start_index: int = 1,
        include_sorry: bool = True,
        attach: bool = False,
    ) -> List[str]:
        if not self.lean_env or not self.lean_env.labels:
            raise ValueError("No Lean labels provided (pass lean_label=... to Graffiti3).")

        if sophie_conditions is None:
            if self.result is None:
                raise ValueError("No sophie conditions provided and self.result is None.")
            sophie_conditions = self.result.sophie_conditions

        out: List[str] = []
        for k, sc in enumerate(sophie_conditions, start=start_index):
            thm = f"{prefix}_{k}"
            s = sophie_condition_to_lean_theorem(
                sc,
                self.lean_env,
                thm,
                base_condition=self.base_property,
                include_sorry=include_sorry,
            )
            out.append(s)
            if attach:
                setattr(sc, "lean_theorem", s)
        return out


# ───────────────── Sophie helpers / pretty printer ───────────────── #



def sophie_runner(
    *,
    conjectures: Sequence[Conjecture],
    g4: Graffiti3,
) -> List[SophieCondition]:
    """
    Thin wrapper around the Sophie module: take a list of inequality
    conjectures and return a flat list of SophieCondition objects.
    """
    if not conjectures:
        return []

    df_num = g4.df
    bool_cols = list(g4.gcr.boolean_cols)
    bool_df = df_num[bool_cols].copy() if bool_cols else pd.DataFrame(index=df_num.index)

    base_mask = g4.base_mask
    base_name = g4.base_name

    sophie_by_prop = discover_sophie_from_inequalities(
        df_num=df_num,
        bool_df=bool_df,
        base_mask=base_mask,
        base_name=base_name,
        inequality_conjectures=conjectures,
        **g4.sophie_cfg,
    )

    flat: List[SophieCondition] = []
    for _, conds in sophie_by_prop.items():
        flat.extend(conds)
    return flat


def _group_sophie_by_property(
    conds: Sequence[SophieCondition],
) -> Dict[str, List[SophieCondition]]:
    """
    Group a flat list of SophieCondition into the dictionary format expected
    by rank_sophie_conditions_global.
    """
    out: Dict[str, List[SophieCondition]] = {}
    for sc in conds:
        out.setdefault(sc.property_name, []).append(sc)
    return out


from textwrap import fill, indent
from typing import Mapping, Optional


def print_g3_result(
    result: Graffiti3Result,
    *,
    k_conjectures: int = 20,
    k_sophie: int = 20,
    data_doc: Optional[Mapping[str, str]] = None,
) -> None:
    """
    Convenience printer for Graffiti3Result: optional dataset/column
    documentation, stage breakdown, top conjectures (sorted by touches),
    and top Sophie conditions.

    Parameters
    ----------
    result : Graffiti3Result
        The result object returned by Graffiti3.run(...).
    k_conjectures : int, optional
        Maximum number of conjectures to print, by default 20.
    k_sophie : int, optional
        Maximum number of Sophie-style conditions to print, by default 20.
    data_doc : Mapping[str, str], optional
        Optional dictionary mapping column names (and the special key
        'data description') to human-readable descriptions. If provided,
        this information is printed before the conjecture lists.
    """

    # ───────── optional dataset / column documentation ───────── #
    if data_doc is not None and len(data_doc) > 0:
        desc = data_doc.get("data description") or data_doc.get("description")

        if desc:
            print("=== Dataset description ===")
            print(fill(str(desc), width=79))
            print()

        # All other keys treated as column descriptions
        column_items = [
            (k, v) for k, v in data_doc.items()
            if k not in {"data description", "description"}
        ]

        if column_items:
            print("=== Column descriptions ===")
            # Keep order if dict is ordered; otherwise, sort by key
            for col, text in column_items:
                print(f"- {col}:")
                wrapped = fill(str(text), width=76)
                print(indent(wrapped, "    "))
                print()

        print("=" * 79)
        print()

    # ───────── Graffiti3 summary and conjectures ───────── #
    print("Stage breakdown:", result.stage_breakdown)
    print(f"Total conjectures: {len(result.conjectures)}")
    print(f"Total Sophie conditions: {len(result.sophie_conditions)}\n")

    print("=== Top conjectures (by touch_count, then support) ===\n")
    for i, c in enumerate(result.conjectures[:k_conjectures], 1):
        touches = getattr(c, "touch_count", getattr(c, "touch", "?"))
        support = getattr(c, "support_n", getattr(c, "support", "?"))
        print(f"Conjecture {i}. {c.pretty()}   [touches={touches}, support={support}]\n")

    print_sophie_conditions(result.sophie_conditions, top_n=k_sophie)
