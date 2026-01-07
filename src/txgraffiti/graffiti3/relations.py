# src/txgraffiti/graffiti3/relations.py
from __future__ import annotations

"""
Generic, DataFrame-agnostic conjecture primitives.

- Relation types (row-wise on a DataFrame): Eq, Le, Ge, AllOf, AnyOf
- Conjecture: (R | C) meaning “for all rows in class C, relation R holds”

Conventions
-----------
- Relation.evaluate(df) -> boolean Series aligned to df.index.
- Relation.slack(df) -> float Series aligned to df.index where >= 0 means satisfied.
  Le:  slack = rhs - lhs
  Ge:  slack = lhs - rhs
  Eq:  slack = tol - |lhs - rhs|
  AllOf: min(child slacks)
  AnyOf: max(child slacks)

- Conjecture.check(df) returns:
    applicable : mask where the condition holds,
    holds      : mask indicating satisfaction of (R | C),
    failures   : df rows with applicable & ~evaluate, plus "__slack__".

User-facing display
-------------------
Conjecture.pretty() yields math-style strings like:
    (planar ∧ regular) ⇒ (alpha ≤ mu) ∧ (alpha ≥ ⌊order/3⌋)
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple, Union, List

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from .exprs import Expr, to_expr, Const, ColumnTerm, LinearForm, BinOp, UnaryOp, LogOp, Func2Op
from .predicates import Predicate, Where, AndPred, OrPred, NotPred, Compare, Between, InSet

__all__ = [
    "Relation",
    "Eq",
    "Le",
    "Ge",
    "AllOf",
    "AnyOf",
    "Conjecture",
    "TRUE",
    "LeanLabel",
    "LeanEnv",
    "conjecture_to_lean_theorem",
]

from math import gcd
from functools import reduce

def _lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b) if a and b else 0

def _lcm_list(ds):
    ds = [d for d in ds if d != 0]
    return reduce(_lcm, ds, 1) if ds else 1

def _scale_rational_terms_to_int(terms):
    """
    terms: list[(q: Fraction, term: Expr)]
    Returns (D, int_terms) where D is the LCM denominator, and int_terms is list[(k:int, term:Expr)]
    representing D * sum(q*term) = sum(k*term).
    """
    denoms = [q.denominator for (q, _) in terms]
    D = _lcm_list(denoms)

    int_terms = []
    for q, t in terms:
        qi = q * D
        # qi should be an integer
        if qi.denominator != 1:
            # shouldn't happen if D is lcm of denominators, but be safe
            return None
        int_terms.append((int(qi.numerator), t))
    return D, int_terms

def _render_int_coeff_combo(int_terms, env: "LeanEnv", ty: str) -> str:
    """
    int_terms: list[(k:int, term:Expr)] representing sum k*term
    ty is one of "ℚ","ℤ","ℝ" (or even "ℕ" if you want).
    """
    from txgraffiti.graffiti3.exprs import Const

    pieces = []
    for k, t in int_terms:
        if k == 0:
            continue

        # constant term: k * 1
        if isinstance(t, Const) and int(t.value) == 1:
            pieces.append(_lean_const(k, ty))
            continue

        t_str = _lean_expr(t, env, ty)

        if k == 1:
            pieces.append(t_str)
        elif k == -1:
            pieces.append(f"(-({t_str}))")
        else:
            k_str = _lean_const(k, ty)
            pieces.append(f"({k_str} * {t_str})")

    if not pieces:
        return f"(0 : {ty})"
    return " + ".join(pieces)

def _move_negative_rhs_to_lhs(L_int_terms, R_int_terms):
    """
    Normalize a linear inequality/equality represented as integer-coefficient term lists.

    Given:
      L = sum(k_i * t_i)
      R = sum(m_j * u_j)

    Move any negative-coefficient terms to the other side so that
    BOTH sides contain only nonnegative coefficients:

      - If a term has k < 0 on the LHS, move it to RHS with coefficient -k.
      - If a term has k < 0 on the RHS, move it to LHS with coefficient -k.

    Returns (L_terms', R_terms').

    This is purely algebraic normalization (works over ℤ/ℚ/ℝ once you've cleared denominators).
    """
    L_pos, L_neg = [], []
    for k, t in L_int_terms:
        (L_pos if k >= 0 else L_neg).append((k, t))

    R_pos, R_neg = [], []
    for k, t in R_int_terms:
        (R_pos if k >= 0 else R_neg).append((k, t))

    # Move negative LHS terms to RHS
    moved_to_R = [(-k, t) for (k, t) in L_neg]  # k < 0 -> -k > 0

    # Move negative RHS terms to LHS
    moved_to_L = [(-k, t) for (k, t) in R_neg]  # k < 0 -> -k > 0

    L_out = L_pos + moved_to_L
    R_out = R_pos + moved_to_R

    # Optional: drop zero coefficients if they exist
    L_out = [(k, t) for (k, t) in L_out if k != 0]
    R_out = [(k, t) for (k, t) in R_out if k != 0]

    return L_out, R_out


# =========================================================
# TRUE predicate (universal class)
# =========================================================

class TRUE_Predicate(Predicate):
    name: str = "TRUE"
    def mask(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(True, index=df.index, dtype=bool)
    def __repr__(self) -> str:
        return "TRUE"

TRUE = TRUE_Predicate()

# =========================================================
# Small pretty helpers
# =========================================================

def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        return s[1:-1].strip()
    return s

def _strip_redundant_parens(s: str) -> str:
    """
    Remove a single pair of outer parentheses when they are syntactically
    redundant, e.g.

        "(radius)"         -> "radius"
        "((order))"        -> "(order)"   (only one layer at a time)
        "(a + b)"          -> "(a + b)"   (kept, has top-level operator)

    We only strip if the entire string is wrapped and there is no
    top-level operator outside inner parentheses.
    """
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return s

    inner = s[1:-1].strip()
    # quick exit: empty or trivial
    if not inner:
        return s

    depth = 0
    for ch in inner:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                # unbalanced; be conservative
                return s
        elif ch in "+-·/%^" and depth == 0:
            # there is a top-level operator; need the parens
            return s

    # No top-level operator found: outer parens are redundant
    return inner


def _pretty_predicate(cond: Predicate, *, unicode_ops: bool = True) -> str:
    """
    Render predicates compactly: strip one outer () if present, then wrap in ().
    """
    s = repr(cond)
    s = _strip_outer_parens(s)
    return f"({s})"

def _balance_neg_terms(L_int, R_int, op: str):
    """
    L_int, R_int: list[(k:int, term:Expr)] possibly with negative k
    op in {"≤","<","≥",">","="}
    Returns (L2, R2) where all coefficients are >= 0 by moving negative terms across.
    Works for ≤ and ≥ (and can be used for <, >, = similarly).
    """
    L_pos, L_neg = [], []
    R_pos, R_neg = [], []

    for k, t in L_int:
        (L_pos if k >= 0 else L_neg).append((k, t))
    for k, t in R_int:
        (R_pos if k >= 0 else R_neg).append((k, t))

    # Move negative terms to the other side: subtracting is not allowed in ℕ, so we add the moved terms.
    # For any inequality form, the safe transformation is:
    #   L_pos - L_neg  op  R_pos - R_neg
    # becomes
    #   L_pos + R_neg  op  R_pos + L_neg
    def flip_sign(ts):
        return [(-k, t) for (k, t) in ts]  # k is negative, so -k is positive

    L2 = [(k, t) for (k, t) in L_pos] + flip_sign(R_neg)
    R2 = [(k, t) for (k, t) in R_pos] + flip_sign(L_neg)

    # drop zeros
    L2 = [(k, t) for (k, t) in L2 if k != 0]
    R2 = [(k, t) for (k, t) in R2 if k != 0]
    return L2, R2

# =========================================================
# Relations
# =========================================================

class Relation:
    """Abstract base: row-wise boolean relation + slack margin."""
    name: str = "Relation"

    # --- core API ---
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def slack(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    # --- helpers and sugar ---
    def is_tight(self, df: pd.DataFrame, *, atol: float = 1e-12) -> pd.Series:
        """
        Rows where the relation is satisfied at equality (boundary), robust to FP error.
        For Le/Ge/Eq this corresponds to slack ≈ 0.
        """
        s = self.slack(df).reindex(df.index)
        return pd.Series(np.isclose(s.values, 0.0, atol=atol), index=s.index, dtype=bool)

    # composition: R1 & R2, R1 | R2
    def __and__(self, other: "Relation") -> "AllOf":
        return AllOf([self, other])

    def __or__(self, other: "Relation") -> "AnyOf":
        return AnyOf([self, other])

    # unified pretty signature (subclasses may override)
    def pretty(self, *, unicode_ops: bool = True, show_tol: bool = False) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return f"{self.name}()"


@dataclass
class Eq(Relation):
    """Equality with absolute tolerance: left == right (within tol)."""
    left: Union[Expr, float, int, str]
    right: Union[Expr, float, int, str]
    tol: float = 1e-9
    name: str = "Equality"

    def __post_init__(self):
        self.left = to_expr(self.left)
        self.right = to_expr(self.right)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.eval(df); r = self.right.eval(df)
        m = np.isclose(l.values, r.values, atol=self.tol)
        return pd.Series(m, index=df.index, dtype=bool)

    def slack(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.eval(df); r = self.right.eval(df)
        return pd.Series(self.tol - np.abs(l - r), index=df.index, dtype=float)

    def pretty(self, *, unicode_ops: bool = True, show_tol: bool = False) -> str:
        eq = "=" if unicode_ops else "=="
        lhs = repr(self.left)
        rhs = repr(self.right)
        if show_tol:
            pm = "±" if unicode_ops else "+/-"
            return f"{lhs} {eq} {rhs} {pm} {self.tol:g}"
        return f"{lhs} {eq} {rhs}"

    def __repr__(self) -> str:
        # Pretty by default (unicode), include tol only when nonzero and helpful
        pm = f" ± {self.tol:g}" if self.tol else ""
        return f"{repr(self.left)} = {repr(self.right)}{pm}"

class Lt(Relation):
    """
    Strict less-than: lhs < rhs - tol.
    If tol==0, this is a pointwise strict <.
    """
    def __init__(self, lhs: Expr, rhs: Union[Expr, float, int], *, tol: float = 0.0, name: Optional[str] = None):
        self.lhs = to_expr(lhs)
        self.rhs = to_expr(rhs)
        self.tol = float(tol)
        self._name = name

    def pretty(self) -> str:
        L = self.lhs.pretty() if hasattr(self.lhs, "pretty") else repr(self.lhs)
        R = self.rhs.pretty() if hasattr(self.rhs, "pretty") else repr(self.rhs)
        if self.tol > 0.0:
            return self._name or f"{L} < {R} - {self.tol:g}"
        return self._name or f"{L} < {R}"

    def evaluate(self, df: pd.DataFrame, condition: Optional["Predicate"] = None) -> pd.Series:
        a = self.lhs.eval(df).astype(float, copy=False)
        b = self.rhs.eval(df).astype(float, copy=False)
        mask = a < (b - self.tol)
        # Ensure boolean series aligned to df; drop NaNs as False
        if hasattr(mask, "fillna"):
            mask = mask.fillna(False)
        mask = mask.astype(bool, copy=False)
        if condition is not None:
            C = condition.mask(df).astype(bool, copy=False)
            mask = mask & C
        return mask

class Gt(Relation):
    """
    Strict greater-than: lhs > rhs + tol.
    If tol==0, this is a pointwise strict >.
    """
    def __init__(self, lhs: Expr, rhs: Union[Expr, float, int], *, tol: float = 0.0, name: Optional[str] = None):
        self.lhs = to_expr(lhs)
        self.rhs = to_expr(rhs)
        self.tol = float(tol)
        self._name = name

    def pretty(self) -> str:
        L = self.lhs.pretty() if hasattr(self.lhs, "pretty") else repr(self.lhs)
        R = self.rhs.pretty() if hasattr(self.rhs, "pretty") else repr(self.rhs)
        if self.tol > 0.0:
            return self._name or f"{L} > {R} + {self.tol:g}"
        return self._name or f"{L} > {R}"

    def evaluate(self, df: pd.DataFrame, condition: Optional["Predicate"] = None) -> pd.Series:
        a = self.lhs.eval(df).astype(float, copy=False)
        b = self.rhs.eval(df).astype(float, copy=False)
        mask = a > (b + self.tol)
        if hasattr(mask, "fillna"):
            mask = mask.fillna(False)
        mask = mask.astype(bool, copy=False)
        if condition is not None:
            C = condition.mask(df).astype(bool, copy=False)
            mask = mask & C
        return mask

@dataclass
class Le(Relation):
    """Inequality: left <= right ; slack = (right - left)."""
    left: Union[Expr, float, int, str]
    right: Union[Expr, float, int, str]
    name: str = "Inequality(<=)"

    def __post_init__(self):
        self.left = to_expr(self.left)
        self.right = to_expr(self.right)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.eval(df); r = self.right.eval(df)
        return pd.Series((l <= r).values, index=df.index, dtype=bool)

    def slack(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.eval(df); r = self.right.eval(df)
        return pd.Series((r - l).values, index=df.index, dtype=float)

    def _lhs_rhs_str(self, unicode_ops: bool = True) -> tuple[str, str]:
        # Use Expr.pretty() when available, then strip redundant outer parens
        if hasattr(self.left, "pretty"):
            lhs = self.left.pretty()
        else:
            lhs = repr(self.left)

        if hasattr(self.right, "pretty"):
            rhs = self.right.pretty()
        else:
            rhs = repr(self.right)

        lhs = _strip_redundant_parens(lhs)
        rhs = _strip_redundant_parens(rhs)
        return lhs, rhs

    def pretty(self, *, unicode_ops: bool = True, show_tol: bool = False) -> str:
        sym = "≤" if unicode_ops else "<="
        lhs, rhs = self._lhs_rhs_str(unicode_ops=unicode_ops)
        return f"{lhs} {sym} {rhs}"

    def __repr__(self) -> str:
        lhs, rhs = self._lhs_rhs_str(unicode_ops=True)
        return f"{lhs} ≤ {rhs}"


@dataclass
class Ge(Relation):
    """Inequality: left >= right ; slack = (left - right)."""
    left: Union[Expr, float, int, str]
    right: Union[Expr, float, int, str]
    name: str = "Inequality(>=)"

    def __post_init__(self):
        self.left = to_expr(self.left)
        self.right = to_expr(self.right)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.eval(df); r = self.right.eval(df)
        return pd.Series((l >= r).values, index=df.index, dtype=bool)

    def slack(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.eval(df); r = self.right.eval(df)
        return pd.Series((l - r).values, index=df.index, dtype=float)

    def _lhs_rhs_str(self, unicode_ops: bool = True) -> tuple[str, str]:
        if hasattr(self.left, "pretty"):
            lhs = self.left.pretty()
        else:
            lhs = repr(self.left)

        if hasattr(self.right, "pretty"):
            rhs = self.right.pretty()
        else:
            rhs = repr(self.right)

        lhs = _strip_redundant_parens(lhs)
        rhs = _strip_redundant_parens(rhs)
        return lhs, rhs

    def pretty(self, *, unicode_ops: bool = True, show_tol: bool = False) -> str:
        sym = "≥" if unicode_ops else ">="
        lhs, rhs = self._lhs_rhs_str(unicode_ops=unicode_ops)
        return f"{lhs} {sym} {rhs}"

    def __repr__(self) -> str:
        lhs, rhs = self._lhs_rhs_str(unicode_ops=True)
        return f"{lhs} ≥ {rhs}"

@dataclass
class AllOf(Relation):
    """Conjunction of relations: R1 ∧ R2 ∧ ... ; slack = min(child slacks)."""
    parts: Iterable[Relation]
    name: str = "AllOf"

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        out = pd.Series(True, index=df.index, dtype=bool)
        for r in self.parts:
            out &= r.evaluate(df).reindex(df.index).astype(bool)
        return out

    def slack(self, df: pd.DataFrame) -> pd.Series:
        slacks: List[pd.Series] = [r.slack(df).reindex(df.index) for r in self.parts]
        if not slacks:
            return pd.Series(0.0, index=df.index, dtype=float)
        return pd.concat(slacks, axis=1).min(axis=1)

    def pretty(self, *, unicode_ops: bool = True, show_tol: bool = False) -> str:
        glue = " ∧ " if unicode_ops else " & "
        items: List[str] = []
        for p in self.parts:
            if hasattr(p, "pretty"):
                items.append(p.pretty(unicode_ops=unicode_ops, show_tol=show_tol))  # type: ignore[call-arg]
            else:
                items.append(repr(p))
        return glue.join(items)

    def __repr__(self) -> str:
        # Use each part's __repr__ (already pretty) and join with ∧
        return " ∧ ".join(repr(p) for p in self.parts)

@dataclass
class AnyOf(Relation):
    """Disjunction of relations: R1 ∨ R2 ∨ ... ; slack = max(child slacks)."""
    parts: Iterable[Relation]
    name: str = "AnyOf"

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        out = pd.Series(False, index=df.index, dtype=bool)
        for r in self.parts:
            out |= r.evaluate(df).reindex(df.index).astype(bool)
        return out

    def slack(self, df: pd.DataFrame) -> pd.Series:
        slacks: List[pd.Series] = [r.slack(df).reindex(df.index) for r in self.parts]
        if not slacks:
            return pd.Series(0.0, index=df.index, dtype=float)
        return pd.concat(slacks, axis=1).max(axis=1)

    def pretty(self, *, unicode_ops: bool = True, show_tol: bool = False) -> str:
        glue = " ∨ " if unicode_ops else " | "
        items: List[str] = []
        for p in self.parts:
            if hasattr(p, "pretty"):
                items.append(p.pretty(unicode_ops=unicode_ops, show_tol=show_tol))  # type: ignore[call-arg]
            else:
                items.append(repr(p))
        return glue.join(items)

    def __repr__(self) -> str:
        # Use each part's __repr__ (already pretty) and join with ∨
        return " ∨ ".join(repr(p) for p in self.parts)


# =========================================================
# Conjecture: (R | C)
# =========================================================

@dataclass
class Conjecture:
    """
    General form: For any object in class C, relation R holds.  (R | C)

    .check(df, auto_base=True) returns:
      - applicable: mask where C holds,
      - holds:      mask for (R | C),
      - failures:   rows of df with applicable & ~evaluate + "__slack__".
    """
    relation: Relation
    condition: Optional[Predicate] = None
    name: str = "Conjecture"
    coefficient_pairs = None
    intercept = None

    # cached for nicer repr/pretty if condition is None and auto_base=True was used
    _resolved_condition: Optional[Predicate] = field(default=None, init=False, repr=False)

    # --------------------------- internals ---------------------------

    def _auto_base(self, df: pd.DataFrame) -> Predicate:
        """
        Detect a base predicate from boolean always-True columns.
        Supports both bool and pandas' nullable BooleanDtype.
        """
        if df is None or df.empty:
            return TRUE

        always_true_cols: List[str] = []
        for col in df.columns:
            s = df[col]
            if is_bool_dtype(s):
                # treat NaN as False for this test
                if bool(pd.Series(s).fillna(False).all()):
                    always_true_cols.append(col)

        if not always_true_cols:
            return TRUE

        preds = [Where(lambda d, c=col: d[c], name=f"{col}") for col in always_true_cols]
        base: Predicate = preds[0]
        for p in preds[1:]:
            base = AndPred(base, p)
        base.name = " ∧ ".join(f"{c}" for c in always_true_cols)
        return base

    # --------------------------- public API ---------------------------

    def check(
        self,
        df: pd.DataFrame,
        *,
        auto_base: bool = True,
    ) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        Evaluate the conjecture on a DataFrame.

        Returns:
            applicable, holds, failures
        """
        # resolve condition
        if self.condition is not None:
            cond = self.condition
        else:
            cond = self._auto_base(df) if auto_base else TRUE

        applicable = cond.mask(df).reindex(df.index, fill_value=False).astype(bool)
        eval_mask = self.relation.evaluate(df).reindex(df.index).astype(bool)
        holds = (~applicable) | (applicable & eval_mask)

        failing = (applicable & ~eval_mask)
        failures = df.loc[failing].copy()
        if failing.any():
            s = self.relation.slack(df).reindex(df.index)
            failures["__slack__"] = s.loc[failing]

        # cache for nicer __repr__/pretty
        self._resolved_condition = cond
        return applicable, holds, failures

    def is_true(self, df: pd.DataFrame, *, auto_base: bool = True) -> bool:
        applicable, holds, _ = self.check(df, auto_base=auto_base)
        return bool(holds[applicable].all())

    def touch_count(self, df: pd.DataFrame, *, auto_base: bool = True, atol: float = 1e-12) -> int:
        applicable, _, _ = self.check(df, auto_base=auto_base)
        tight = self.relation.is_tight(df, atol=atol).reindex(df.index)
        val = int((applicable & tight).sum())
        # keep both for backward compatibility
        setattr(self, "touch", val)
        setattr(self, "touch_count", val)
        return val

    def violation_count(self, df: pd.DataFrame, *, auto_base: bool = True) -> int:
        applicable, holds, _ = self.check(df, auto_base=auto_base)
        return int((applicable & ~holds).sum())

    def pretty(
        self,
        arrow: Optional[str] = None,
        *,
        unicode_ops: bool = True,
        show_tol: bool = False,
    ) -> str:
        """
        Human-facing rendering:
            (cond) ⇒ (lhs ≤ rhs) ∧ (lhs ≥ rhs)  ...
        If the condition is TRUE, returns just the relation string.

        Parameters
        ----------
        arrow : Optional[str]
            Force the arrow symbol. Defaults to '⇒' if unicode_ops else '->'.
        unicode_ops : bool
            Use unicode math symbols.
        show_tol : bool
            If True, show ±tol for Eq relations.
        """
        cond = self.condition or self._resolved_condition or TRUE

        if hasattr(self.relation, "pretty"):
            rel_str = self.relation.pretty(unicode_ops=unicode_ops, show_tol=show_tol)  # type: ignore[call-arg]
        else:
            rel_str = repr(self.relation)

        # If TRUE, omit condition.
        if isinstance(cond, TRUE_Predicate):
            return rel_str

        cond_str = _pretty_predicate(cond, unicode_ops=unicode_ops)
        arr = ("⇒" if unicode_ops else "->") if arrow is None else arrow
        return f"{cond_str} {arr} {rel_str}"

    def signature(self) -> str:
        """Canonical-ish string signature for deduplication (mirrors pretty())."""
        return self.pretty(unicode_ops=True, show_tol=False)


    def to_lean_theorem(
        self,
        env: "LeanEnv",
        theorem_name: str,
        *,
        base_condition: Optional[Predicate] = None,
        include_sorry: bool = True,
    ) -> str:
        """Return a Lean4 theorem stub for this conjecture."""
        return conjecture_to_lean_theorem(
            self,
            env,
            theorem_name,
            base_condition=base_condition,
            include_sorry=include_sorry,
        )

    def __repr__(self) -> str:
        # Keep a compact debug form that is still pretty—same as pretty() w/o arrow
        cond = self.condition or self._resolved_condition or TRUE
        if isinstance(cond, TRUE_Predicate):
            return f"{repr(self.relation)}"
        return f"Conjecture({repr(self.relation)} | {repr(cond)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Conjecture):
            return False
        # Use the canonical signature (pretty + normalized condition)
        return self.signature() == other.signature()

    def __hash__(self) -> int:
        # Avoid recursion; hash a stable, human-readable signature
        return hash(self.signature())


# -----------------------------
# Helpers
# -----------------------------

def _bool_mask(p: Predicate, df: pd.DataFrame) -> pd.Series:
    """Mask from a predicate, aligned, dtype=bool, NA->False."""
    m = p.mask(df).reindex(df.index, fill_value=False)
    if m.dtype != bool:
        m = m.fillna(False).astype(bool, copy=False)
    return m

def _pred_name(p: Predicate) -> str:
    """
    Compact display name for a predicate.
    Uses repr(p) but strips redundant outer parens like '((planar))' -> '(planar)'.
    """
    s = repr(p).strip()
    # normalize any accidental double-wrapping
    while len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        inner = s[1:-1].strip()
        # stop if removing would change grouping (look for bare ' ∧ ', ' ∨ ', etc.)
        # but since our Predicate.__repr__ already adds its own parens sensibly,
        # one layer strip is enough for tidiness.
        s = inner
        break
    return f"({s})"

# =========================================================
# Lean4 rendering
# =========================================================
from dataclasses import dataclass as _dc_dataclass, field as _dc_field
from typing import Any as _Any, Dict as _Dict, Iterable as _Iterable, List as _List, Optional as _Optional, Tuple as _Tuple, Union as _Union
from fractions import Fraction as _Fraction
from typing import Any as _Any, Dict as _Dict, Iterable as _Iterable, List as _List, Optional as _Optional, Set as _Set, Tuple as _Tuple
import re

_NUM_TYPE_RANK = {"ℕ": 0, "Nat": 0, "ℤ": 1, "Int": 1, "ℚ": 2, "Rat": 2, "ℝ": 3, "Real": 3}

@_dc_dataclass(frozen=True)
class LeanLabel:
    """How a dataset column/property is written in Lean."""
    term: str                  # e.g. "independence_number" or "connected"
    type: _Optional[str] = None  # e.g. "ℕ", "ℚ", "ℝ", or "Prop"
    kind: _Optional[str] = None  # "invariant" | "predicate" | "binder" | None

@_dc_dataclass
class LeanEnv:
    """
    Renderer context.

    `labels` maps dataset column names -> LeanLabel.

    Special keys in lean_label mapping:
      - "__object__": ("G", "SimpleGraph V")  (backward-compatible)
      - "__binders__": [("n","ℕ"), ("A","Matrix (Fin m) (Fin n) ℝ"), "(m n : ℕ)", ...]
      - "__primary__": 0  (index of the binder that auto-application targets; default 0)
      - "__defaults__": {"num_type": "ℝ"}
    """
    labels: _Dict[str, LeanLabel]

    # General signature support
    binders: _List[_Union[str, _Tuple[str, str]]] = _dc_field(default_factory=list)
    primary: int = 0

    # Backward-compatible single-object fields
    obj_name: str = "G"
    obj_type: str = "SimpleGraph V"

    default_num_type: str = "ℝ"

    @staticmethod
    def from_mapping(m: _Optional[_Dict[str, _Any]]) -> "LeanEnv":
        if not m:
            return LeanEnv(labels={})

        obj_name = "G"
        obj_type = "SimpleGraph V"
        default_num_type = "ℝ"
        binders = []
        primary = 0

        labels: _Dict[str, LeanLabel] = {}

        def _as_label(v: _Any) -> LeanLabel:
            if isinstance(v, LeanLabel):
                return v
            if isinstance(v, tuple) and len(v) == 2:
                return LeanLabel(term=str(v[0]), type=str(v[1]))
            if isinstance(v, dict):
                return LeanLabel(
                    term=str(v.get("term", "")),
                    type=(str(v["type"]) if "type" in v and v["type"] is not None else None),
                    kind=(str(v["kind"]) if "kind" in v and v["kind"] is not None else None),
                )
            return LeanLabel(term=str(v), type=None)

        # Backward-compatible __object__
        if "__object__" in m:
            ov = m["__object__"]
            if isinstance(ov, dict):
                obj_name = str(ov.get("name", obj_name))
                obj_type = str(ov.get("type", obj_type))
            elif isinstance(ov, (tuple, list)) and len(ov) == 2:
                obj_name, obj_type = str(ov[0]), str(ov[1])

        # New: __binders__
        if "__binders__" in m and isinstance(m["__binders__"], list):
            for b in m["__binders__"]:
                if isinstance(b, str):
                    binders.append(b)  # raw binder chunk, e.g. "(m n : ℕ)" or "[Fintype V]"
                elif isinstance(b, (tuple, list)) and len(b) == 2:
                    binders.append((str(b[0]), str(b[1])))
            # if the first binder is a (name,type) pair, use it as the default obj for auto-application
            for item in binders:
                if isinstance(item, tuple):
                    obj_name, obj_type = item[0], item[1]
                    break

        # Which binder should term auto-application target?
        if "__primary__" in m:
            try:
                primary = int(m["__primary__"])
            except Exception:
                primary = 0

        if "__defaults__" in m and isinstance(m["__defaults__"], dict):
            default_num_type = str(m["__defaults__"].get("num_type", default_num_type))

        for k, v in m.items():
            if str(k).startswith("__"):
                continue
            labels[str(k)] = _as_label(v)

        env = LeanEnv(
            labels=labels,
            binders=binders,
            primary=primary,
            obj_name=obj_name,
            obj_type=obj_type,
            default_num_type=default_num_type,
        )
        return env

    def binder(self) -> str:
        # If user provided __binders__, render those; otherwise fall back to single-object binder
        if self.binders:
            parts = []
            for b in self.binders:
                if isinstance(b, str):
                    parts.append(b)
                else:
                    name, ty = b
                    parts.append(f"({name} : {ty})")
            return " ".join(parts)
        return f"({self.obj_name} : {self.obj_type})"

    def _primary_obj_name(self) -> str:
        if not self.binders:
            return self.obj_name
        # pick nth tuple binder; if primary points at a string binder, fall back to first tuple binder
        tuple_binders = [b for b in self.binders if isinstance(b, tuple)]
        if not tuple_binders:
            return self.obj_name
        idx = self.primary
        if 0 <= idx < len(tuple_binders):
            return tuple_binders[idx][0]
        return tuple_binders[0][0]

    def _apply_to_obj(self, term: str) -> str:
        # Support {obj}, {obj0}, {obj1}, ... placeholders
        fmt = {"obj": self._primary_obj_name()}
        tuple_binders = [b for b in self.binders if isinstance(b, tuple)]
        for i, (nm, _ty) in enumerate(tuple_binders):
            fmt[f"obj{i}"] = nm

        if "{" in term and "}" in term:
            try:
                term = term.format(**fmt)
            except Exception:
                pass

        # If the term already mentions the primary object name, keep it
        obj = fmt["obj"]
        if re.search(rf"\b{re.escape(obj)}\b", term):
            return term

        # If term is a bare identifier, apply it to primary object
        if term and re.match(r"^[A-Za-z_][A-Za-z0-9_'.]*$", term):
            return f"{term} {obj}"

        return term

    def term_of(self, col: str) -> str:
        lbl = self.labels.get(col)
        if not lbl:
            return col
        return self._apply_to_obj(lbl.term)

    def type_of(self, col: str) -> _Optional[str]:
        lbl = self.labels.get(col)
        return lbl.type if lbl else None


    def _norm_num_type(self, ty: str) -> str:
        """Normalize common numeric type spellings to the canonical symbols you use elsewhere."""
        ty = str(ty).strip()
        aliases = {
            "Nat": "ℕ", "ℕ": "ℕ",
            "Int": "ℤ", "ℤ": "ℤ",
            "Rat": "ℚ", "ℚ": "ℚ",
            "Real": "ℝ", "ℝ": "ℝ",
        }
        return aliases.get(ty, ty)

    def join_num_type(self, tys) -> str:
        """
        Join numeric Lean types by promoting to the 'largest' (ℕ < ℤ < ℚ < ℝ).
        If nothing is provided, fall back to env.default_num_type.
        """
        order = {"ℕ": 0, "ℤ": 1, "ℚ": 2, "ℝ": 3}

        best = None
        best_rank = -1
        for t in tys:
            if not t:
                continue
            t = self._norm_num_type(t)
            r = order.get(t, None)
            if r is None:
                # Unknown numeric type: be conservative and prefer default_num_type if set
                # (or treat as Real by default).
                r = order.get(self.default_num_type, 3)
                t = self.default_num_type
            if r > best_rank:
                best_rank = r
                best = t

        return best if best is not None else self.default_num_type

    def obj_decl(self) -> str:
        """
        Backward-compatible: return the single primary object binder.
        Used by older Sophie/theorem emitters expecting "(G : SimpleGraph V)".
        """
        # If we have explicit binders, return the first *tuple* binder if present
        if self.binders:
            for b in self.binders:
                if isinstance(b, tuple):
                    name, ty = b
                    return f"({name} : {ty})"
            # Otherwise, fall back to the legacy object fields
        return f"({self.obj_name} : {self.obj_type})"

    def obj_name_type(self) -> tuple[str, str]:
        """
        Backward-compatible helper: return (name, type) of the primary object.
        """
        if self.binders:
            for b in self.binders:
                if isinstance(b, tuple):
                    return b[0], b[1]
        return self.obj_name, self.obj_type

def _lean_const(value: _Any, ty: str) -> str:
    if isinstance(value, bool):
        return f"({str(value).lower()} : {ty})"
    if isinstance(value, _Fraction):
        n, d = int(value.numerator), int(value.denominator)
        if d == 1:
            return f"({n} : {ty})"
        return f"(({n} : {ty}) / ({d} : {ty}))"

    if isinstance(value, (int,)):
        return f"({int(value)} : {ty})"
    try:
        import numpy as _np
        if isinstance(value, (_np.integer,)):
            return f"({int(value)} : {ty})"
        if isinstance(value, (_np.floating, float)):
            frac = _Fraction(float(value)).limit_denominator(10**6)
            if frac.denominator == 1:
                return f"({int(frac.numerator)} : {ty})"
            return f"(({int(frac.numerator)} : {ty}) / ({int(frac.denominator)} : {ty}))"
    except Exception:
        pass
    return f"({repr(value)} : {ty})"

def _expr_cols(e: Expr, out: set[str]) -> None:
    """Collect column names referenced by an Expr tree.

    Matches the Expr model in exprs.py:
      - ColumnTerm.col
      - LinearForm.terms
      - BinOp.left/right
      - UnaryOp.arg
      - LogOp.arg
      - Func2Op.left/right
    """
    from txgraffiti.graffiti3.exprs import (
        Const, ColumnTerm, LinearForm, BinOp, UnaryOp, LogOp, Func2Op, Expr
    )

    if isinstance(e, ColumnTerm):
        out.add(e.col)
        return

    if isinstance(e, LinearForm):
        for _, c in e.terms:
            out.add(c)
        return

    if isinstance(e, BinOp):
        _expr_cols(e.left, out)
        _expr_cols(e.right, out)
        return

    if isinstance(e, UnaryOp):
        _expr_cols(e.arg, out)
        return

    if isinstance(e, LogOp):
        _expr_cols(e.arg, out)
        return

    if isinstance(e, Func2Op):
        _expr_cols(e.left, out)
        _expr_cols(e.right, out)
        return

    if isinstance(e, Const):
        return

    # Fallback for forward-compatibility
    for attr in ("arg", "left", "right", "a", "b"):
        if hasattr(e, attr):
            v = getattr(e, attr)
            if isinstance(v, Expr):
                _expr_cols(v, out)

def _infer_num_type_from_expr(e: Expr, env: LeanEnv) -> str:
    cols: _Set[str] = set()
    _expr_cols(e, cols)
    tys = [env.type_of(c) for c in cols]
    tys2 = [t for t in tys if t]
    return env.join_num_type(tys2) if tys2 else env.default_num_type

def _as_nat_exponent(e: "Expr") -> int | None:
    """If e is a constant representing a nonnegative integer, return it, else None."""
    from fractions import Fraction as _Fraction
    from txgraffiti.graffiti3.exprs import Const
    try:
        import numpy as _np
        np_integer = (_np.integer,)
        np_floating = (_np.floating,)
    except Exception:
        np_integer = tuple()
        np_floating = tuple()

    if not isinstance(e, Const):
        return None

    v = e.value

    # Fraction like 2/1
    if isinstance(v, _Fraction):
        if v.denominator == 1 and v.numerator >= 0:
            return int(v.numerator)
        return None

    # int-like
    if isinstance(v, (int,) + np_integer):
        return int(v) if int(v) >= 0 else None

    # float-like close to integer
    if isinstance(v, (float,) + np_floating):
        r = round(float(v))
        if abs(float(v) - r) < 1e-12 and r >= 0:
            return int(r)

    return None

def _lean_expr(e: object, env: "LeanEnv", ty: str) -> str:
    """Render an Expr (or scalar) as a Lean term of type `ty`."""
    from txgraffiti.graffiti3.exprs import (
        Const, ColumnTerm, LinearForm, BinOp, UnaryOp, LogOp, Func2Op, Expr, to_expr
    )
    import numpy as np

    e = to_expr(e) if not isinstance(e, Expr) else e

    # ---- constants / columns ----
    if isinstance(e, Const):
        return _lean_const(e.value, ty)

    if isinstance(e, ColumnTerm):
        term = env.term_of(e.col)
        return f"({term} : {ty})"

    # ---- linear form: intercept + Σ coef*col ----
    if isinstance(e, LinearForm):
        parts: list[str] = []
        parts.append(_lean_const(e.intercept, ty))
        for coef, col in e.terms:
            a = _lean_const(coef, ty)
            x = f"({env.term_of(col)} : {ty})"
            parts.append(f"{a} * {x}")
        return f"({' + '.join(parts)})"

    # ---- binary ops ----
    if isinstance(e, BinOp):
        op = e.fn
        L = _lean_expr(e.left, env, ty)
        R = _lean_expr(e.right, env, ty)

        if op is np.add or getattr(op, "__name__", "") == "add":
            return f"({L} + {R})"
        if op is np.subtract or getattr(op, "__name__", "") == "subtract":
            return f"({L} - {R})"
        if op is np.multiply or getattr(op, "__name__", "") == "multiply":
            return f"({L} * {R})"
        if op is np.divide or getattr(op, "__name__", "") == "divide":
            return f"({L} / {R})"

        # numpy power -> Lean pow / rpow
        if op is np.power or getattr(op, "__name__", "") == "power":
            n = _as_nat_exponent(e.right)
            if n is not None:
                return f"({L} ^ ({n} : ℕ))"
            # fallback: real exponent -> Real.rpow (only sensible in ℝ)
            baseR = L if ty == "ℝ" else f"({L} : ℝ)"
            expR = _lean_expr(e.right, env, "ℝ")
            return f"(Real.rpow {baseR} {expR})"

        op_name = getattr(op, "__name__", "binop")
        return f"({op_name} {L} {R})"

    # ---- unary ops ----
    if isinstance(e, UnaryOp):
        op = e.fn

        # Special-case abs FIRST because we need access to the raw argument structure
        if op is np.abs or getattr(op, "__name__", "") in {"absolute", "abs"}:
            inner = e.arg

            # Nat: abs(a - b) should become Nat.dist a b (compilable, correct)
            if ty == "ℕ":
                if isinstance(inner, BinOp) and (
                    inner.fn is np.subtract or getattr(inner.fn, "__name__", "") == "subtract"
                ):
                    L = _lean_expr(inner.left, env, "ℕ")
                    R = _lean_expr(inner.right, env, "ℕ")
                    return f"(Nat.dist {L} {R})"

                # Fallback: lift to ℤ and take natAbs (ensures compilation)
                Z = _lean_expr(inner, env, "ℤ")
                return f"(Int.natAbs {Z})"

            # Non-Nat: standard absolute value notation
            A = _lean_expr(inner, env, ty)
            return f"|{A}|"

        # Other unary ops can use the rendered argument
        A = _lean_expr(e.arg, env, ty)

        if op is np.negative or getattr(op, "__name__", "") == "negative":
            return f"(-{A})"

        # sqrt: assume it lives in ℝ
        if op is np.sqrt or getattr(op, "__name__", "") in {"safe_sqrt_series", "safe_sqrt", "sqrt"}:
            if ty == "ℝ":
                return f"(Real.sqrt {A})"
            return f"(Real.sqrt ({A} : ℝ))"

        if op is np.exp or getattr(op, "__name__", "") == "exp":
            if ty == "ℝ":
                return f"(Real.exp {A})"
            return f"(Real.exp ({A} : ℝ))"

        op_name = getattr(op, "__name__", "unary")
        return f"({op_name} {A})"

    # ---- log ops ----
    if isinstance(e, LogOp):
        # Note: this always lives in ℝ in your Lean rendering
        argR = _lean_expr(e.arg, env, "ℝ")
        if e.epsilon and float(e.epsilon) > 0.0:
            eps = _lean_const(float(e.epsilon), "ℝ")
            argR = f"(max {argR} {eps})"

        if e.base is None:
            return f"(Real.log {argR})"

        # Change of base: log a / log b
        base = float(e.base)
        baseR = _lean_const(base, "ℝ")
        return f"((Real.log {argR}) / (Real.log {baseR}))"

    # ---- binary functions (min/max) ----
    if isinstance(e, Func2Op):
        L = _lean_expr(e.left, env, ty)
        R = _lean_expr(e.right, env, ty)
        fn = e.fn
        name = getattr(fn, "__name__", "")
        if fn is np.minimum or name == "minimum":
            return f"(min {L} {R})"
        if fn is np.maximum or name == "maximum":
            return f"(max {L} {R})"
        fn_name = getattr(fn, "__name__", "func2")
        return f"({fn_name} {L} {R})"

    # ---- fallback ----
    return str(e)


# # def _lean_expr(e: object, env: "LeanEnv", ty: str) -> str:
# #     """Render an Expr (or scalar) as a Lean term of type `ty`."""
# #     from txgraffiti.graffiti3.exprs import (
# #         Const, ColumnTerm, LinearForm, BinOp, UnaryOp, LogOp, Func2Op, Expr, to_expr
# #     )
# #     import numpy as np

# #     e = to_expr(e) if not isinstance(e, Expr) else e

# #     if isinstance(e, Const):
# #         return _lean_const(e.value, ty)

# #     if isinstance(e, ColumnTerm):
# #         term = env.term_of(e.col)
# #         return f"({term} : {ty})"

# #     if isinstance(e, LinearForm):
# #         parts: list[str] = []
# #         parts.append(_lean_const(e.intercept, ty))
# #         for coef, col in e.terms:
# #             a = _lean_const(coef, ty)
# #             x = f"({env.term_of(col)} : {ty})"
# #             parts.append(f"{a} * {x}")
# #         return f"({' + '.join(parts)})"

# #     if isinstance(e, BinOp):
# #         op = e.fn
# #         L = _lean_expr(e.left, env, ty)
# #         R = _lean_expr(e.right, env, ty)

# #         if op is np.add:
# #             return f"({L} + {R})"
# #         if op is np.subtract:
# #             return f"({L} - {R})"
# #         if op is np.multiply:
# #             return f"({L} * {R})"
# #         if op is np.divide:
# #             return f"({L} / {R})"

# #         # ✅ IMPORTANT: numpy power -> Lean pow
# #         if op is np.power:
# #             n = _as_nat_exponent(e.right)
# #             if n is not None:
# #                 # exponent as Nat
# #                 return f"({L} ^ ({n} : ℕ))"
# #             # fallback: real exponent -> Real.rpow (only sensible in ℝ)
# #             baseR = L if ty == "ℝ" else f"({L} : ℝ)"
# #             expR = _lean_expr(e.right, env, "ℝ")
# #             return f"(Real.rpow {baseR} {expR})"

# #         op_name = getattr(op, "__name__", "binop")
# #         return f"({op_name} {L} {R})"


# #     if isinstance(e, UnaryOp):
# #         op = e.fn
# #         A = _lean_expr(e.arg, env, ty)

# #         if op is np.abs:
# #             return f"(|{A}|)"

# #         if op is np.negative:
# #             return f"(-{A})"

# #         # sqrt: assume it lives in ℝ if it was generated
# #         if op is np.sqrt or getattr(op, "__name__", "") in {"safe_sqrt_series", "safe_sqrt"}:
# #             if ty == "ℝ":
# #                 return f"(Real.sqrt {A})"
# #             return f"(Real.sqrt ({A} : ℝ))"

# #         if op is np.exp:
# #             if ty == "ℝ":
# #                 return f"(Real.exp {A})"
# #             return f"(Real.exp ({A} : ℝ))"

# #         op_name = getattr(op, "__name__", "unary")
# #         return f"({op_name} {A})"

#     if isinstance(e, LogOp):
#         # log stages should infer ℝ; force ℝ rendering here
#         A = _lean_expr(e.arg, env, "ℝ")
#         if e.base is None:
#             return f"(Real.log {A})"
#         b = _lean_const(e.base, "ℝ")
#         return f"((Real.log {A}) / (Real.log {b}))"

#     if isinstance(e, Func2Op):
#         L = _lean_expr(e.left, env, ty)
#         R = _lean_expr(e.right, env, ty)
#         nm = e.name
#         if nm in {"min", "max"}:
#             return f"({nm} {L} {R})"
#         return f"({nm} {L} {R})"

#     if isinstance(e, str):
#         return f"({env.term_of(e)} : {ty})"

#     return _lean_const(e, ty)

def _flatten_and(p: Predicate) -> _List[Predicate]:
    if isinstance(p, AndPred):
        return _flatten_and(p.a) + _flatten_and(p.b)
    return [p]


def _lean_pred(p: Predicate, env: LeanEnv) -> str:
    if isinstance(p, AndPred):
        return f"({_lean_pred(p.a, env)} ∧ {_lean_pred(p.b, env)})"
    if isinstance(p, OrPred):
        return f"({_lean_pred(p.a, env)} ∨ {_lean_pred(p.b, env)})"
    if isinstance(p, NotPred):
        return f"(¬ {_lean_pred(p.a, env)})"
    if isinstance(p, Compare):
        tys: _List[str] = []
        for side in (p.left, p.right):
            if isinstance(side, str):
                t = env.type_of(side)
                if t:
                    tys.append(t)
            elif isinstance(side, Expr):
                tys.append(_infer_num_type_from_expr(side, env))
        ty = env.join_num_type(tys) if tys else env.default_num_type

        op_map = {"<=": "≤", ">=": "≥", "<": "<", ">": ">", "==": "=", "!=": "≠"}
        sym = op_map.get(p._symbol, p._symbol)
        L = _lean_expr(p.left, env, ty)
        R = _lean_expr(p.right, env, ty)
        return f"{L} {sym} {R}"

    if isinstance(p, Where):
        raw = getattr(p, "name", None) or repr(p)
        s = str(raw).strip()
        if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
            s = s[1:-1].strip()
        term = env.term_of(s)
        return term

    if isinstance(p, Between):
        ty = env.default_num_type
        x = _lean_expr(p.x, env, ty)
        lo = _lean_expr(p.lo, env, ty)
        hi = _lean_expr(p.hi, env, ty)
        return f"({lo} ≤ {x} ∧ {x} ≤ {hi})"
    if isinstance(p, InSet):
        term = env.term_of(str(p.x)) if isinstance(p.x, str) else _lean_expr(p.x, env, env.default_num_type)
        return f"({term} ∈ {p.S})"

    return str(p)


def _rel_operands(r: Relation):
    """Return (L, R) operands for any Relation variant."""
    if hasattr(r, "left") and hasattr(r, "right"):
        return r.left, r.right
    if hasattr(r, "lhs") and hasattr(r, "rhs"):
        return r.lhs, r.rhs
    return None, None

def _render_rational_combo(terms, env: "LeanEnv", ty: str) -> str:
    """
    terms: list[(q: Fraction, term: Expr)] representing sum q*term
    """
    from txgraffiti.graffiti3.exprs import Const

    pieces = []
    for q, t in terms:
        if q == 0:
            continue

        # constant term represented as q * 1
        if isinstance(t, Const) and int(t.value) == 1:
            pieces.append(_lean_const(q, ty))
            continue

        t_str = _lean_expr(t, env, ty)

        if q == 1:
            pieces.append(f"{t_str}")
        elif q == -1:
            # keep as unary negation instead of (-1)*t
            pieces.append(f"(-({t_str}))")
        else:
            q_str = _lean_const(q, ty)
            pieces.append(f"({q_str} * {t_str})")

    if not pieces:
        return f"(0 : {ty})"
    return " + ".join(pieces)

# def _lean_relation(r: Relation, env: LeanEnv) -> str:
#     # Atomic comparisons
#     if isinstance(r, (Le, Lt, Ge, Gt, Eq)):
#         L_raw, R_raw = _rel_operands(r)
#         if L_raw is None or R_raw is None:
#             return str(r)

#         ty = env.join_num_type(
#             filter(
#                 None,
#                 [
#                     _infer_num_type_from_expr(L_raw, env)
#                     if isinstance(L_raw, Expr)
#                     else (env.type_of(L_raw) if isinstance(L_raw, str) else None),
#                     _infer_num_type_from_expr(R_raw, env)
#                     if isinstance(R_raw, Expr)
#                     else (env.type_of(R_raw) if isinstance(R_raw, str) else None),
#                 ],
#             )
#         )

#         # --- ℕ path: clear denominators (and your balancing happens inside _clear_denoms_nat) ---
#         if ty == "ℕ" and (_expr_has_fraction(L_raw) or _expr_has_fraction(R_raw)):
#             cleared = _clear_denoms_nat(L_raw, R_raw, env)
#             if cleared is not None:
#                 L, R = cleared
#                 if isinstance(r, Le): return f"{L} ≤ {R}"
#                 if isinstance(r, Lt): return f"{L} < {R}"
#                 if isinstance(r, Ge): return f"{L} ≥ {R}"
#                 if isinstance(r, Gt): return f"{L} > {R}"
#                 if isinstance(r, Eq): return f"{L} = {R}"
#             # else: fall through

#         # --- ℚ/ℤ/ℝ path: move negative RHS terms to LHS (your preference) ---
#         # Only do this for the common "linear combo" shapes we know how to render nicely.
#         if ty in {"ℕ", "ℚ", "ℤ", "ℝ"}:
#             try:
#                 L_terms = _rational_linear_terms(L_raw)
#                 R_terms = _rational_linear_terms(R_raw)

#                 # split RHS into positive and negative coefficients
#                 R_pos, R_neg = [], []
#                 for q, t in R_terms:
#                     (R_pos if q >= 0 else R_neg).append((q, t))

#                 # move all negative RHS terms to LHS
#                 if R_neg:
#                     L_terms = L_terms + [(-q, t) for (q, t) in R_neg]
#                     R_terms = R_pos

#                 L = _render_rational_combo(L_terms, env, ty)
#                 R = _render_rational_combo(R_terms, env, ty)

#                 if isinstance(r, Le): return f"{L} ≤ {R}"
#                 if isinstance(r, Lt): return f"{L} < {R}"
#                 if isinstance(r, Ge): return f"{L} ≥ {R}"
#                 if isinstance(r, Gt): return f"{L} > {R}"
#                 if isinstance(r, Eq): return f"{L} = {R}"
#             except Exception:
#                 # if anything unexpected happens, fall back to old rendering
#                 pass

#         # --- fallback: old rendering ---
#         L = _lean_expr(L_raw, env, ty)
#         R = _lean_expr(R_raw, env, ty)

#         if isinstance(r, Le): return f"{L} ≤ {R}"
#         if isinstance(r, Lt): return f"{L} < {R}"
#         if isinstance(r, Ge): return f"{L} ≥ {R}"
#         if isinstance(r, Gt): return f"{L} > {R}"
#         if isinstance(r, Eq): return f"{L} = {R}"

#     # Boolean combos
#     if isinstance(r, AllOf):
#         return " ∧ ".join(f"({_lean_relation(p, env)})" for p in r.parts)
#     if isinstance(r, AnyOf):
#         return " ∨ ".join(f"({_lean_relation(p, env)})" for p in r.parts)

#     return str(r)

def _expr_is_linear_rational_no_log(e) -> bool:
    """
    True if e can be safely handled by _rational_linear_terms as a linear combo
    with rational scalar coefficients, AND contains no LogOp.
    """
    if _expr_has_log(e):
        return False
    # If you want: exclude other non-linear ops too (sqrt, exp, etc.)
    # For now: rely on _rational_linear_terms behavior + try/except in caller.
    return True


def _expr_has_log(e) -> bool:
    e = to_expr(e) if not isinstance(e, Expr) else e
    if isinstance(e, LogOp):
        return True
    if isinstance(e, BinOp):
        return _expr_has_log(e.left) or _expr_has_log(e.right)
    if isinstance(e, UnaryOp):
        return _expr_has_log(getattr(e, "arg", None))
    if isinstance(e, Func2Op):
        return _expr_has_log(e.left) or _expr_has_log(e.right)
    return False

def _expr_has_sqrt(e) -> bool:
    from txgraffiti.graffiti3.exprs import Expr, to_expr, Const, ColumnTerm, LinearForm, BinOp, UnaryOp, LogOp, Func2Op
    import numpy as np

    e = to_expr(e) if not isinstance(e, Expr) else e

    if isinstance(e, (Const, ColumnTerm, LinearForm)):
        return False

    if isinstance(e, UnaryOp):
        name = getattr(e.fn, "__name__", "")
        if e.fn is np.sqrt or name in {"sqrt", "safe_sqrt_series", "safe_sqrt"}:
            return True
        return _expr_has_sqrt(e.arg)

    if isinstance(e, LogOp):
        return _expr_has_sqrt(e.arg)

    if isinstance(e, BinOp):
        return _expr_has_sqrt(e.left) or _expr_has_sqrt(e.right)

    if isinstance(e, Func2Op):
        return _expr_has_sqrt(e.left) or _expr_has_sqrt(e.right)

    return False

def _expr_has_real_forcing_op(e) -> bool:
    """
    True if rendering e will necessarily introduce Real.* operators
    (log/sqrt/exp/rpow), meaning the surrounding relation must be typed in ℝ.
    """
    from txgraffiti.graffiti3.exprs import Expr, to_expr, Const, ColumnTerm, LinearForm, BinOp, UnaryOp, LogOp, Func2Op
    import numpy as np

    e = to_expr(e) if not isinstance(e, Expr) else e

    if isinstance(e, (Const, ColumnTerm, LinearForm)):
        return False

    if isinstance(e, LogOp):
        return True  # always rendered with Real.log

    if isinstance(e, UnaryOp):
        name = getattr(e.fn, "__name__", "")
        if e.fn is np.sqrt or name in {"sqrt", "safe_sqrt_series", "safe_sqrt"}:
            return True  # rendered with Real.sqrt
        if e.fn is np.exp or name == "exp":
            return True  # rendered with Real.exp
        return _expr_has_real_forcing_op(e.arg)

    if isinstance(e, BinOp):
        # rpow happens when power exponent isn't a Nat literal
        if e.fn is np.power or getattr(e.fn, "__name__", "") == "power":
            n = _as_nat_exponent(e.right)
            if n is None:
                return True  # rendered as Real.rpow
        return _expr_has_real_forcing_op(e.left) or _expr_has_real_forcing_op(e.right)

    if isinstance(e, Func2Op):
        return _expr_has_real_forcing_op(e.left) or _expr_has_real_forcing_op(e.right)

    return False


# def _lean_relation(r: Relation, env: LeanEnv) -> str:
#     # Atomic comparisons
#     if isinstance(r, (Le, Lt, Ge, Gt, Eq)):
#         L_raw, R_raw = _rel_operands(r)
#         if L_raw is None or R_raw is None:
#             return str(r)

#         ty = env.join_num_type(
#             filter(
#                 None,
#                 [
#                     _infer_num_type_from_expr(L_raw, env)
#                     if isinstance(L_raw, Expr)
#                     else (env.type_of(L_raw) if isinstance(L_raw, str) else None),
#                     _infer_num_type_from_expr(R_raw, env)
#                     if isinstance(R_raw, Expr)
#                     else (env.type_of(R_raw) if isinstance(R_raw, str) else None),
#                 ],
#             )
#         )

#         # ---- ℕ: clear denominators (and your Nat balancing is inside _clear_denoms_nat) ----
#         if ty == "ℕ" and (_expr_has_fraction(L_raw) or _expr_has_fraction(R_raw)):
#             cleared = _clear_denoms_nat(L_raw, R_raw, env)
#             if cleared is not None:
#                 L, R = cleared
#                 if isinstance(r, Le): return f"{L} ≤ {R}"
#                 if isinstance(r, Lt): return f"{L} < {R}"
#                 if isinstance(r, Ge): return f"{L} ≥ {R}"
#                 if isinstance(r, Gt): return f"{L} > {R}"
#                 if isinstance(r, Eq): return f"{L} = {R}"
#             # else: fall through

#         # ---- ℚ/ℤ/ℝ: clear denominators AND move negative RHS terms to LHS ----
#         if ty in {"ℚ", "ℤ", "ℝ"}:
#             try:
#                 # collect as sum(q*term)
#                 L_terms = _rational_linear_terms(L_raw)
#                 R_terms = _rational_linear_terms(R_raw)

#                 # scale both sides by common LCM denominator so all coeffs become integers
#                 scaled_L = _scale_rational_terms_to_int(L_terms)
#                 scaled_R = _scale_rational_terms_to_int(R_terms)

#                 if scaled_L is not None and scaled_R is not None:
#                     DL, L_int = scaled_L
#                     DR, R_int = scaled_R

#                     # Use a single D that works for both sides (LCM of the two Ds)
#                     D = _lcm(DL, DR)
#                     if D != DL:
#                         mul = D // DL
#                         L_int = [(k * mul, t) for (k, t) in L_int]
#                     if D != DR:
#                         mul = D // DR
#                         R_int = [(k * mul, t) for (k, t) in R_int]

#                     # optional gcd reduction (keeps things smaller)
#                     coeffs = [abs(k) for (k, _) in (L_int + R_int) if k != 0]
#                     g = reduce(gcd, coeffs, 0)
#                     if g and g > 1:
#                         L_int = [(k // g, t) for (k, t) in L_int]
#                         R_int = [(k // g, t) for (k, t) in R_int]

#                     # move negative RHS terms to LHS
#                     L_int, R_int = _move_negative_rhs_to_lhs(L_int, R_int)

#                     L = _render_int_coeff_combo(L_int, env, ty)
#                     R = _render_int_coeff_combo(R_int, env, ty)

#                     if isinstance(r, Le): return f"{L} ≤ {R}"
#                     if isinstance(r, Lt): return f"{L} < {R}"
#                     if isinstance(r, Ge): return f"{L} ≥ {R}"
#                     if isinstance(r, Gt): return f"{L} > {R}"
#                     if isinstance(r, Eq): return f"{L} = {R}"
#             except Exception:
#                 pass  # fall through

#         # ---- fallback: old rendering ----
#         L = _lean_expr(L_raw, env, ty)
#         R = _lean_expr(R_raw, env, ty)

#         if isinstance(r, Le): return f"{L} ≤ {R}"
#         if isinstance(r, Lt): return f"{L} < {R}"
#         if isinstance(r, Ge): return f"{L} ≥ {R}"
#         if isinstance(r, Gt): return f"{L} > {R}"
#         if isinstance(r, Eq): return f"{L} = {R}"

#     # Boolean combos
#     if isinstance(r, AllOf):
#         return " ∧ ".join(f"({_lean_relation(p, env)})" for p in r.parts)
#     if isinstance(r, AnyOf):
#         return " ∨ ".join(f"({_lean_relation(p, env)})" for p in r.parts)

#     return str(r)

def _lean_relation(r: Relation, env: LeanEnv) -> str:
    # Atomic comparisons
    if isinstance(r, (Le, Lt, Ge, Gt, Eq)):
        L_raw, R_raw = _rel_operands(r)
        if L_raw is None or R_raw is None:
            return str(r)

        ty = env.join_num_type(
            filter(
                None,
                [
                    _infer_num_type_from_expr(L_raw, env)
                    if isinstance(L_raw, Expr)
                    else (env.type_of(L_raw) if isinstance(L_raw, str) else None),
                    _infer_num_type_from_expr(R_raw, env)
                    if isinstance(R_raw, Expr)
                    else (env.type_of(R_raw) if isinstance(R_raw, str) else None),
                ],
            )
        )

        # Force ℝ if either side contains LogOp (since _lean_expr(LogOp, ...) uses Real.log)
        # if _expr_has_log(L_raw) or _expr_has_log(R_raw):
        #     ty = "ℝ"
        if _expr_has_log(L_raw) or _expr_has_log(R_raw) or _expr_has_sqrt(L_raw) or _expr_has_sqrt(R_raw):
            ty = "ℝ"

        if _expr_has_real_forcing_op(L_raw) or _expr_has_real_forcing_op(R_raw):
            ty = "ℝ"

        # ---- ℕ: clear denominators (and Nat balancing is inside _clear_denoms_nat) ----
        if ty == "ℕ" and (_expr_has_fraction(L_raw) or _expr_has_fraction(R_raw)):
            cleared = _clear_denoms_nat(L_raw, R_raw, env)
            if cleared is not None:
                L, R = cleared
                if isinstance(r, Le): return f"{L} ≤ {R}"
                if isinstance(r, Lt): return f"{L} < {R}"
                if isinstance(r, Ge): return f"{L} ≥ {R}"
                if isinstance(r, Gt): return f"{L} > {R}"
                if isinstance(r, Eq): return f"{L} = {R}"
            # else fall through

        # ---- ℚ/ℤ/ℝ: clear denominators AND move negative RHS terms to LHS ----
        # This is safe even when terms contain LogOp, because the coefficients are still rationals
        # and LogOp just becomes a "term" in the linear combo.
        if ty in {"ℕ", "ℚ", "ℤ", "ℝ"}:
            try:
                L_terms = _rational_linear_terms(L_raw)
                R_terms = _rational_linear_terms(R_raw)

                scaled_L = _scale_rational_terms_to_int(L_terms)
                scaled_R = _scale_rational_terms_to_int(R_terms)

                if scaled_L is not None and scaled_R is not None:
                    DL, L_int = scaled_L
                    DR, R_int = scaled_R

                    # Use a single D that works for both sides (LCM of the two Ds)
                    D = _lcm(DL, DR)
                    if D != DL:
                        mul = D // DL
                        L_int = [(k * mul, t) for (k, t) in L_int]
                    if D != DR:
                        mul = D // DR
                        R_int = [(k * mul, t) for (k, t) in R_int]

                    # optional gcd reduction (keeps things smaller)
                    coeffs = [abs(k) for (k, _) in (L_int + R_int) if k != 0]
                    g = reduce(gcd, coeffs, 0)
                    if g and g > 1:
                        L_int = [(k // g, t) for (k, t) in L_int]
                        R_int = [(k // g, t) for (k, t) in R_int]

                    # move negative RHS terms to LHS (your preference)
                    L_int, R_int = _move_negative_rhs_to_lhs(L_int, R_int)

                    L = _render_int_coeff_combo(L_int, env, ty)
                    R = _render_int_coeff_combo(R_int, env, ty)

                    if isinstance(r, Le): return f"{L} ≤ {R}"
                    if isinstance(r, Lt): return f"{L} < {R}"
                    if isinstance(r, Ge): return f"{L} ≥ {R}"
                    if isinstance(r, Gt): return f"{L} > {R}"
                    if isinstance(r, Eq): return f"{L} = {R}"
            except Exception:
                pass  # fall through

        # ---- fallback: old rendering ----
        L = _lean_expr(L_raw, env, ty)
        R = _lean_expr(R_raw, env, ty)

        if isinstance(r, Le): return f"{L} ≤ {R}"
        if isinstance(r, Lt): return f"{L} < {R}"
        if isinstance(r, Ge): return f"{L} ≥ {R}"
        if isinstance(r, Gt): return f"{L} > {R}"
        if isinstance(r, Eq): return f"{L} = {R}"

    # Boolean combos (optional: remove extra parentheses)
    if isinstance(r, AllOf):
        return " ∧ ".join(_lean_relation(p, env) for p in r.parts)
    if isinstance(r, AnyOf):
        return " ∨ ".join(_lean_relation(p, env) for p in r.parts)

    return str(r)



def _hyp_name_for(stmt: str, used: _Set[str]) -> str:
    candidates = []
    # if "connected" in stmt:
    #     candidates.append("h_conn")
    # if "order" in stmt and ("≥" in stmt or ">=" in stmt):
    #     candidates.append("h_min")
    candidates.append("h1")
    candidates.extend([f"h{i}" for i in range(2, 1000)])

    for c in candidates:
        if c not in used:
            used.add(c)
            return c
    c = f"h{len(used)+1}"
    used.add(c)
    return c


# def conjecture_to_lean_theorem(
#     conj: "Conjecture",
#     env: LeanEnv,
#     theorem_name: str,
#     *,
#     base_condition: _Optional[Predicate] = None,
#     include_sorry: bool = True,
# ) -> str:
#     """Render a Conjecture as a Lean4 theorem stub.

#     This is deliberately best-effort and purely syntactic; it assumes your `env.labels`
#     can translate your dataset columns into Lean names/types.
#     """
#     if env is None:
#         raise ValueError("LeanEnv is required to render Lean code.")

#     cond = conj.condition or conj._resolved_condition or base_condition or TRUE

#     hyps: _List[str] = []
#     if not isinstance(cond, TRUE_Predicate):
#         atoms = _flatten_and(cond)
#         seen: _Set[str] = set()
#         for a in atoms:
#             s = _lean_pred(a, env)
#             if s not in seen:
#                 seen.add(s)
#                 hyps.append(s)

#     used_names: _Set[str] = set()
#     hyp_lines = ""
#     for h in hyps:
#         hn = _hyp_name_for(h, used_names)
#         hyp_lines += f"\n    ({hn} : {h})"

#     goal = _lean_relation(conj.relation, env)

#     body = "sorry" if include_sorry else "by\n  sorry"

#     return (
#         f"theorem {theorem_name} {env.binder()}"
#         f"{hyp_lines}\n  : {goal} :=\n{body}"
#     )
# -----------------------------
# Class inclusion:  A ⊆ B
# -----------------------------

@dataclass
class ClassInclusion:
    """
    Logical class inclusion: ``A ⊆ B`` (i.e., implication ``A → B`` holds row-wise).

    Methods
    -------
    mask(df)            : (~A) | B
    violations(df)      : rows with A & ~B
    violation_count(df) : count of violations
    holds_all(df)       : mask(df).all()
    pretty(...)         : "(A) ⊆ (B)" (or ASCII: "(A) <= (B)")
    signature()         : stable pretty string
    """
    A: Predicate
    B: Predicate
    name: str = "ClassInclusion"

    # --- core ---

    def mask(self, df: pd.DataFrame) -> pd.Series:
        a = _bool_mask(self.A, df)
        b = _bool_mask(self.B, df)
        return (~a) | b

    def violations(self, df: pd.DataFrame) -> pd.DataFrame:
        a = _bool_mask(self.A, df)
        b = _bool_mask(self.B, df)
        bad = a & ~b
        return df.loc[bad]

    # --- handy helpers ---

    def violation_count(self, df: pd.DataFrame) -> int:
        a = _bool_mask(self.A, df)
        b = _bool_mask(self.B, df)
        return int((a & ~b).sum())

    def holds_all(self, df: pd.DataFrame) -> bool:
        """True iff inclusion holds for every row (vacuous where A is False)."""
        return bool(self.mask(df).all())

    # --- formatting ---

    def pretty(self, *, unicode_ops: bool = True) -> str:
        op = "⊆" if unicode_ops else "<="
        return f"{_pred_name(self.A)} {op} {_pred_name(self.B)}"

    def signature(self) -> str:
        return self.pretty(unicode_ops=True)

    def __repr__(self) -> str:
        return f"({self.A!r} ⊆ {self.B!r})"


# -----------------------------
# Class equivalence:  A ≡ B
# -----------------------------

@dataclass
class ClassEquivalence:
    """
    Logical class equivalence: ``A ≡ B`` (row-wise equality of masks).

    Methods
    -------
    mask(df)            : A == B
    violations(df)      : rows where A ^ B
    violation_count(df) : count of violations
    holds_all(df)       : mask(df).all()
    pretty(...)         : "(A) ≡ (B)" (or ASCII: "(A) == (B)")
    signature()         : stable pretty string
    """
    A: Predicate
    B: Predicate
    name: str = "ClassEquivalence"

    # --- core ---

    def mask(self, df: pd.DataFrame) -> pd.Series:
        a = _bool_mask(self.A, df)
        b = _bool_mask(self.B, df)
        return a == b

    def violations(self, df: pd.DataFrame) -> pd.DataFrame:
        a = _bool_mask(self.A, df)
        b = _bool_mask(self.B, df)
        bad = a ^ b
        return df.loc[bad]

    # --- handy helpers ---

    def violation_count(self, df: pd.DataFrame) -> int:
        a = _bool_mask(self.A, df)
        b = _bool_mask(self.B, df)
        return int((a ^ b).sum())

    def holds_all(self, df: pd.DataFrame) -> bool:
        return bool(self.mask(df).all())

    # --- formatting ---

    def pretty(self, *, unicode_ops: bool = True) -> str:
        op = "≡" if unicode_ops else "=="
        return f"{_pred_name(self.A)} {op} {_pred_name(self.B)}"

    def signature(self) -> str:
        return self.pretty(unicode_ops=True)

    def __repr__(self) -> str:
        return f"({self.A!r} ≡ {self.B!r})"

# lean_emit.py
import numpy as np
from dataclasses import dataclass

from txgraffiti.graffiti3.exprs import Const, ColumnTerm, LinearForm, BinOp, UnaryOp, LogOp, Func2Op
from txgraffiti.graffiti3.predicates import AndPred, OrPred, NotPred, Where, Compare, Between, InSet
from txgraffiti.graffiti3.relations import Ge, Le, Eq, AllOf, AnyOf, Conjecture

_BIN = {
    np.add: "+",
    np.subtract: "-",
    np.multiply: "*",
    np.divide: "/",
    np.power: "^",
}

# --- add near the Lean rendering helpers in relations.py ---

from math import gcd
from functools import reduce

def _lcm(a: int, b: int) -> int:
    return abs(a*b) // gcd(a, b) if a and b else 0

def _lcm_list(ds):
    ds = [d for d in ds if d != 0]
    return reduce(_lcm, ds, 1) if ds else 1

def _expr_has_fraction(e) -> bool:
    e = to_expr(e) if not isinstance(e, Expr) else e

    if isinstance(e, Const):
        return isinstance(e.value, _Fraction)

    if isinstance(e, ColumnTerm):
        return False

    if isinstance(e, BinOp):
        return _expr_has_fraction(e.left) or _expr_has_fraction(e.right)

    if isinstance(e, UnaryOp):
        # exprs.py uses .arg; keep fallback for any older naming
        inner = getattr(e, "arg", getattr(e, "a", None))
        return _expr_has_fraction(inner)

    if isinstance(e, LogOp):
        inner = getattr(e, "arg", getattr(e, "a", None))
        return _expr_has_fraction(inner)

    if isinstance(e, Func2Op):
        # exprs.py uses .left/.right; fallback for any older naming
        l = getattr(e, "left", getattr(e, "a", None))
        r = getattr(e, "right", getattr(e, "b", None))
        return _expr_has_fraction(l) or _expr_has_fraction(r)

    if isinstance(e, LinearForm):
        # LinearForm is numeric coefficients + columns; it will not contain Const(Fraction)
        return False

    return False


def _as_fraction_const(e):
    # returns Fraction if e is a Const(Fraction/int-like), else None
    from txgraffiti.graffiti3.exprs import Const, Expr, to_expr
    e = to_expr(e) if not isinstance(e, Expr) else e
    if isinstance(e, Const):
        v = e.value
        if isinstance(v, _Fraction):
            return v
        if isinstance(v, int):
            return _Fraction(v, 1)
    return None

def _rational_linear_terms(e):
    """
    Return list[(coeff: Fraction, term: Expr)] representing e as sum coeff*term
    for the common shapes: add/sub, scalar-mul by Const(Fraction/int), and constants.
    """
    from txgraffiti.graffiti3.exprs import Const, BinOp, Expr, to_expr
    import numpy as np

    e = to_expr(e) if not isinstance(e, Expr) else e

    c = _as_fraction_const(e)
    if c is not None:
        # constant c = c * 1
        return [(c, Const(1))]

    if isinstance(e, BinOp):
        fn = e.fn
        name = getattr(fn, "__name__", "")

        if fn is np.add or name == "add":
            return _rational_linear_terms(e.left) + _rational_linear_terms(e.right)

        if fn is np.subtract or name == "subtract":
            left = _rational_linear_terms(e.left)
            right = _rational_linear_terms(e.right)
            return left + [(-q, t) for (q, t) in right]

        if fn is np.multiply or name == "multiply":
            cl = _as_fraction_const(e.left)
            cr = _as_fraction_const(e.right)
            if cl is not None:
                return [(cl*q, t) for (q, t) in _rational_linear_terms(e.right)]
            if cr is not None:
                return [(cr*q, t) for (q, t) in _rational_linear_terms(e.left)]
            # non-scalar product: treat as a single term
            return [(_Fraction(1,1), e)]

        # other BinOps (power, div, etc): treat as a single term
        return [(_Fraction(1,1), e)]

    # default: treat as a single term
    return [(_Fraction(1,1), e)]

def _render_int_combo(terms, env: "LeanEnv") -> str:
    """
    terms: list[(k: int, term: Expr)] with k >= 0 in Nat mode
    """
    from txgraffiti.graffiti3.exprs import Const
    pieces = []
    for k, t in terms:
        if k == 0:
            continue
        # absorb *1
        if isinstance(t, Const) and int(t.value) == 1:
            pieces.append(f"({k} : ℕ)")
            continue
        t_str = _lean_expr(t, env, "ℕ")
        if k == 1:
            pieces.append(f"{t_str}")
        else:
            pieces.append(f"(({k} : ℕ) * {t_str})")
    if not pieces:
        return "(0 : ℕ)"
    return " + ".join(pieces)

def _clear_denoms_nat(L_raw, R_raw, env: "LeanEnv"):
    """
    Build denominator-cleared Nat expressions for L_raw and R_raw.
    Returns (L_str, R_str).
    """
    L_terms = _rational_linear_terms(L_raw)
    R_terms = _rational_linear_terms(R_raw)

    denoms = [q.denominator for (q, _) in (L_terms + R_terms)]
    D = _lcm_list(denoms)

    # scale and convert to integer coefficients
    def scale(terms):
        out = []
        for q, t in terms:
            qi = q * D
            assert qi.denominator == 1
            out.append((int(qi.numerator), t))
        return out

    L_int = scale(L_terms)
    R_int = scale(R_terms)

    # OPTIONAL reduction by gcd of all coefficients (and D was already folded in)
    all_coeffs = [k for k, _ in (L_int + R_int) if k != 0]
    g = reduce(gcd, all_coeffs, 0)
    if g and g > 1:
        L_int = [(k//g, t) for k, t in L_int]
        R_int = [(k//g, t) for k, t in R_int]

    L_int, R_int = _balance_neg_terms(L_int, R_int, op="≥")  # op doesn’t actually matter here
    # now all coefficients are >= 0
    return _render_int_combo(L_int, env), _render_int_combo(R_int, env)


    # return _render_int_combo(L_int, env), _render_int_combo(R_int, env)


def _op_symbol(sym: str) -> str:
    return {"<=": "≤", ">=": "≥", "!=": "≠"}.get(sym, sym)

@dataclass
class LeanEmitter:
    lean_label: dict[str, str]

    def col(self, key: str) -> str:
        # key = internal name like "radius" / "connected"
        try:
            return self.lean_label[key]
        except KeyError:
            # fallback: still usable while you grow the map
            return f"{key} G"

    # ---------- Expr ----------
    def emit_expr(self, e) -> str:
        if isinstance(e, ColumnTerm):
            return self.col(e.col)

        if isinstance(e, Const):
            # keep it simple; your columns already carry types/casts
            return str(e.pretty())

        if isinstance(e, LinearForm):
            # Safe expansion; keeps you off repr-parsing
            parts = []
            if float(e.intercept) != 0.0:
                parts.append(str(Const(e.intercept).pretty()))
            for a, c in e.terms:
                t = self.col(c)
                if a == 1:
                    parts.append(t)
                elif a == -1:
                    parts.append(f"-({t})")
                else:
                    parts.append(f"({Const(a).pretty()}) * ({t})")
            return " + ".join(parts) if parts else "0"

        if isinstance(e, BinOp):
            op = _BIN.get(e.fn, None)
            L = self.emit_expr(e.left)
            R = self.emit_expr(e.right)
            return f"({L}) {op} ({R})" if op else f"({L}) /*?*/ ({R})"

        if isinstance(e, UnaryOp):
            A = self.emit_expr(e.arg)
            # Start minimal; add cases as you need them
            if e.fn is np.abs:
                return f"abs ({A})"
            if e.fn is np.sqrt:
                return f"Real.sqrt ({A})"
            if e.fn is np.exp:
                return f"Real.exp ({A})"
            return f"/*unary*/ ({A})"

        if isinstance(e, LogOp):
            A = self.emit_expr(e.arg)
            if e.base is None:
                return f"Real.log ({A})"
            return f"(Real.log ({A}) / Real.log ({e.base}))"

        if isinstance(e, Func2Op):
            L = self.emit_expr(e.left)
            R = self.emit_expr(e.right)
            # assume e.name is "min" or "max"
            return f"{e.name} ({L}) ({R})"

        return e.pretty()

    # ---------- Predicate ----------
    def _flatten_and(self, p):
        if isinstance(p, AndPred):
            return self._flatten_and(p.a) + self._flatten_and(p.b)
        return [p]

    def emit_pred(self, p) -> str:
        if isinstance(p, Where):
            # where.name should be "connected", "tree", ...
            return self.col(p.name)

        if isinstance(p, Compare):
            # Prefer structural if possible:
            L = self.emit_expr(p.left) if hasattr(p, "left") else None
            R = self.emit_expr(p.right) if hasattr(p, "right") else None
            sym = _op_symbol(getattr(p, "_symbol", ""))
            if L is not None and R is not None and sym:
                return f"({L}) {sym} ({R})"
            # fallback: use displayed name
            return _op_symbol(p.name)

        if isinstance(p, Between):
            X = self.emit_expr(p.x)
            lo = self.emit_expr(p.low)
            hi = self.emit_expr(p.high)
            return f"({lo}) ≤ ({X}) ∧ ({X}) ≤ ({hi})"

        if isinstance(p, InSet):
            X = self.emit_expr(p.col)
            vs = ", ".join(map(str, list(p.values)))
            return f"({X}) ∈ {{{vs}}}"

        if isinstance(p, NotPred):
            return f"¬ ({self.emit_pred(p.a)})"
        if isinstance(p, OrPred):
            return f"({self.emit_pred(p.a)}) ∨ ({self.emit_pred(p.b)})"
        if isinstance(p, AndPred):
            return f"({self.emit_pred(p.a)}) ∧ ({self.emit_pred(p.b)})"

        return repr(p)

    # ---------- Relation ----------
    def emit_rel(self, r) -> str:
        if isinstance(r, Ge):
            return f"({self.emit_expr(r.left)}) ≥ ({self.emit_expr(r.right)})"
        if isinstance(r, Le):
            return f"({self.emit_expr(r.left)}) ≤ ({self.emit_expr(r.right)})"
        if isinstance(r, Eq):
            return f"({self.emit_expr(r.left)}) = ({self.emit_expr(r.right)})"
        if isinstance(r, AllOf):
            return " ∧ ".join(f"({self.emit_rel(x)})" for x in r.rels) or "True"
        if isinstance(r, AnyOf):
            return " ∨ ".join(f"({self.emit_rel(x)})" for x in r.rels) or "False"
        return repr(r)

    # ---------- Conjecture → theorem ----------
    def emit_theorem(self, c: Conjecture, name: str, default='G : SimpleGraph V') -> str:
        hyps = []
        if c.condition is not None:
            atoms = self._flatten_and(c.condition)
            # dedupe by rendered text
            seen = set()
            for a in atoms:
                s = self.emit_pred(a)
                if s not in seen:
                    hyps.append(s)
                    seen.add(s)

        hyp_lines = "\n  ".join([f"(h{i+1} : {h})" for i, h in enumerate(hyps)])
        goal = self.emit_rel(c.relation)

        return f"""theorem {name} ({default})
  {hyp_lines}
  : {goal} :=
by
  sorry
"""

def _lean_prop_label(prop: str, env: "LeanEnv") -> str:
    s = prop.strip()
    if "&" in s:
        parts = [p.strip() for p in s.split("&") if p.strip()]
        return "(" + " ∧ ".join(_lean_prop_label(p, env) for p in parts) + ")"
    if s.startswith("¬"):
        inner = s[1:].strip()
        return f"(¬ {_lean_prop_label(inner, env)})"
    return f"({env.term_of(s)})"

def sophie_condition_to_lean_theorem(
    sc: "SophieCondition",
    env: "LeanEnv",
    theorem_name: str,
    *,
    base_condition: "Predicate | None" = None,
    include_sorry: bool = True,
) -> str:
    # General binder list (works for graphs, integers, matrices, multi-binders)
    obj = env.binder()

    binders: list[str] = []
    if base_condition is not None:
        binders.extend(_lean_binders_from_predicate(base_condition, env))

    def _is_trivial_true_binder(b: str) -> bool:
        """
        Drop binders like:
          (h1 : True)
          (h1 : (True))
          (h1 : True A)    -- bad/meaningless and won't typecheck anyway
          (h1 : TRUE n)    -- legacy
        """
        s = b.strip()

        # Expect "(hX : <Prop>)"
        prop = s
        if s.startswith("(") and s.endswith(")"):
            inner = s[1:-1].strip()
            parts = inner.split(":", 1)
            prop = parts[1].strip() if len(parts) == 2 else inner.strip()

        # Strip ONE layer of parens around the prop
        while prop.startswith("(") and prop.endswith(")"):
            prop2 = prop[1:-1].strip()
            # stop if we didn't actually remove meaningful parentheses
            if prop2 == prop:
                break
            prop = prop2

        # Now check for True / TRUE, possibly applied
        # Cases:
        #   "True"
        #   "True A"
        #   "True (A)"
        #   "TRUE", "TRUE A", ...
        if prop == "True" or prop == "TRUE":
            return True
        if prop.startswith("True ") or prop.startswith("TRUE "):
            return True
        if prop.startswith("True(") or prop.startswith("TRUE("):
            return True

        return False

    # Drop trivial True binders from base_condition
    binders = [b for b in binders if not _is_trivial_true_binder(b)]

    # Always include Sophie hypothesis
    binders.append(f"(hH : {_lean_relation(sc.hyp_relation, env)})")

    goal = _lean_prop_label(sc.property_name, env)

    header = f"theorem {theorem_name} {obj}"
    binder_block = ("\n    " + "\n    ".join(binders)) if binders else ""
    body = "sorry" if include_sorry else "by\n  sorry"
    return f"{header}{binder_block}\n  : {goal} :=\n{body}"


def conjecture_to_lean_theorem(
    conj: "Conjecture",
    env: LeanEnv,
    theorem_name: str,
    *,
    base_condition: _Optional[Predicate] = None,
    include_sorry: bool = True,
) -> str:
    """Render a Conjecture as a Lean4 theorem stub (best-effort syntactic)."""
    if env is None:
        raise ValueError("LeanEnv is required to render Lean code.")

    # Choose condition (fall back to TRUE-like predicate)
    cond = conj.condition or getattr(conj, "_resolved_condition", None) or base_condition or TRUE

    def _is_trivial_true_prop(prop_str: str) -> bool:
        """
        Returns True if the rendered Lean proposition is trivial True/TRUE (or mis-rendered True applied).
        We drop such binders rather than emitting (h : TRUE).
        """
        s = (prop_str or "").strip()

        # strip outer parentheses repeatedly
        while s.startswith("(") and s.endswith(")"):
            s2 = s[1:-1].strip()
            if s2 == s:
                break
            s = s2

        # Accept both spellings (legacy)
        if s in {"True", "TRUE"}:
            return True

        # Also drop accidental applications that will not typecheck / are meaningless
        # e.g. "True A", "TRUE n", "True (A)"
        if s.startswith("True ") or s.startswith("TRUE "):
            return True
        if s.startswith("True(") or s.startswith("TRUE("):
            return True

        return False

    # Build hypothesis prop strings
    hyps: _List[str] = []
    atoms = []
    try:
        atoms = _flatten_and(cond)
    except Exception:
        atoms = [cond] if cond is not None else []

    seen: _Set[str] = set()
    for a in atoms:
        try:
            s = _lean_pred(a, env)
        except Exception:
            continue

        # Drop trivial True/TRUE hypotheses
        if _is_trivial_true_prop(s):
            continue

        if s and s not in seen:
            seen.add(s)
            hyps.append(s)

    # Render hypothesis lines with names
    used_names: _Set[str] = set()
    hyp_lines = ""
    for h in hyps:
        hn = _hyp_name_for(h, used_names)
        hyp_lines += f"\n    ({hn} : {h})"

    goal = _lean_relation(conj.relation, env)
    body = "sorry" if include_sorry else "by\n  sorry"

    return (
        f"theorem {theorem_name} {env.binder()}"
        f"{hyp_lines}\n  : {goal} :=\n{body}"
    )


def _flatten_predicate_terms(p) -> list:
    """
    Best-effort flattening: assumes your Predicate objects support:
      - And(p, q) via attributes `.left`/`.right` or `.a`/`.b`
      - Atom predicate via `.col` or `.name` or `.pred`
    If your actual predicate classes differ, this still usually works
    because we fall back to str(p).
    """
    # Avoid importing predicate classes to keep this file decoupled.
    if p is None:
        return []
    # common And shapes
    for (lattr, rattr) in (("left", "right"), ("a", "b")):
        if hasattr(p, lattr) and hasattr(p, rattr):
            return _flatten_predicate_terms(getattr(p, lattr)) + _flatten_predicate_terms(getattr(p, rattr))
    return [p]

def _pred_to_lean_prop(p, env) -> str:
    def normalize_key(s: str) -> str:
        s = str(s).strip()
        # strip one layer of outer parentheses
        if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
            s = s[1:-1].strip()
        return s

    # If predicate comes in as a string key, map it through env
    if isinstance(p, str):
        return env.term_of(normalize_key(p))

    # If it’s a predicate object with a column/key attribute
    for attr in ("col", "name", "pred", "key"):
        if hasattr(p, attr):
            k = getattr(p, attr)
            if isinstance(k, str):
                return env.term_of(normalize_key(k))

    # fallback
    return str(p)

def _lean_binders_from_predicate(base_condition, env) -> list[str]:
    """
    Turn a (possibly conjunctive) base_condition into Lean hypothesis binders.
    Produces names h1, h2, ... with a special-case for connected/order≥2 if present.
    """
    terms = _flatten_predicate_terms(base_condition)

    binders: list[str] = []
    used: set[str] = set()

    # name heuristics
    def mk_name(prop: str, idx: int) -> str:
        # if "connected" in prop:
        #     return "h_conn"
        # if "order" in prop and "≥" in prop:
        #     return "h_min"
        return f"h{idx}"

    i = 1
    for t in terms:
        prop = _pred_to_lean_prop(t, env)

        # de-dup identical props
        if prop in used:
            continue
        used.add(prop)

        binders.append(f"({mk_name(prop, i)} : {prop})")
        i += 1

    return binders
