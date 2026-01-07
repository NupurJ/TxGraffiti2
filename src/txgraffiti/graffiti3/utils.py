# src/txgraffiti/graffiti3/utils.py

from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from txgraffiti.graffiti3.relations import Conjecture

def _filter_by_touch(
    df: pd.DataFrame,
    conjectures: Sequence[Conjecture],
    min_touches: int,
) -> List[Conjecture]:
    """
    Compute touch_count for each conjecture and discard those with
    touch_count < min_touches.

    This is intended to be used *before* expensive heuristics
    (Morgan, Dalmatian), so that they only see reasonably “tight”
    candidates.
    """
    if min_touches <= 0:
        # Nothing to filter; still ensure touch_count is computed once.
        out: List[Conjecture] = []
        for c in conjectures:
            try:
                c.touch_count(df)
            except Exception:
                setattr(c, "touch_count", 0)
            out.append(c)
        return out

    kept: List[Conjecture] = []
    for c in conjectures:
        try:
            t = c.touch_count(df)  # sets c.touch and c.touch_count
        except Exception:
            t = 0
            setattr(c, "touch_count", 0)
        if t >= min_touches:
            kept.append(c)
    return kept


def _dedup_conjectures(conjs: Sequence[Conjecture]) -> List[Conjecture]:
    """
    Stable dedup by conjecture.signature().

    Keeps the first occurrence of each signature and drops later duplicates.
    """
    seen: set[str] = set()
    out: List[Conjecture] = []
    for c in conjs:
        sig = c.signature()
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
    return out


def _nice_fraction(
    x: float,
    *,
    max_denom: int = 50,
    max_numer: int = 200,
) -> Optional[Fraction]:
    """
    Approximate x by a "nice" rational p/q with small numerator/denominator.

    Returns None if:
      - x is not finite, or
      - |p| > max_numer or q > max_denom.

    This is what prevents coefficients like 4740631186705785/8 from appearing.
    """
    if not np.isfinite(x):
        return None

    frac = Fraction(x).limit_denominator(max_denom)
    if abs(frac.numerator) > max_numer or abs(frac.denominator) > max_denom:
        return None
    return frac


# def _annotate_and_sort_conjectures(
#     df: pd.DataFrame,
#     conjs: Sequence[Conjecture],
# ) -> List[Conjecture]:
#     """
#     Compute touch_count and support_n for each conjecture, deduplicate by
#     signature, and sort by (touch_count, support_n) descending.
#     """
#     unique: List[Conjecture] = []
#     seen: set[str] = set()

#     for c in conjs:
#         sig = c.signature()
#         if sig in seen:
#             continue
#         seen.add(sig)

#         # Compute touch_count once; Conjecture.touch_count mutates itself
#         touch_attr = getattr(c, "touch_count", None)
#         if callable(touch_attr):
#             try:
#                 val = c.touch_count(df, auto_base=False)
#             except TypeError:
#                 # Fallback if signature differs
#                 val = c.touch_count(df)
#         else:
#             # Already materialized as an int
#             val = touch_attr if isinstance(touch_attr, int) else 0

#         setattr(c, "touch_count", int(val))
#         setattr(c, "touch", int(val))  # for backward compatibility

#         # Compute support_n: how many rows are in the hypothesis class
#         try:
#             applicable, _, _ = c.check(df, auto_base=False)
#             support = int(applicable.sum())
#         except Exception:
#             support = 0

#         setattr(c, "support_n", support)
#         setattr(c, "support", support)

#         unique.append(c)

#     unique.sort(
#         key=lambda cc: (
#             int(getattr(cc, "touch_count", 0)),
#             int(getattr(cc, "support_n", 0)),
#         ),
#         reverse=True,
#     )
#     return unique

# def _annotate_and_sort_conjectures(
#     df: pd.DataFrame,
#     conjs: Sequence[Conjecture],
# ) -> List[Conjecture]:
#     """
#     Compute touch_count and support_n for each conjecture, optionally upgrade
#     non-strict inequalities (≤/≥) to equations when touch == support, then
#     deduplicate by signature, and sort by (touch_count, support_n) descending.
#     """
#     # Local import to avoid circular imports at module import time
#     from txgraffiti.graffiti3.relations import Le, Ge, Eq

#     unique: List[Conjecture] = []
#     seen: set[str] = set()

#     for c in conjs:
#         # -----------------------------
#         # Compute touch_count (tight rows)
#         # -----------------------------
#         touch_attr = getattr(c, "touch_count", None)
#         if callable(touch_attr):
#             try:
#                 val = c.touch_count(df, auto_base=False)
#             except TypeError:
#                 val = c.touch_count(df)
#             except Exception:
#                 val = 0
#         else:
#             val = touch_attr if isinstance(touch_attr, int) else 0

#         val = int(val)
#         setattr(c, "touch_count", val)
#         setattr(c, "touch", val)  # backward compatibility

#         # -----------------------------
#         # Compute support_n (applicable rows)
#         # -----------------------------
#         try:
#             applicable, _, _ = c.check(df, auto_base=False)
#             support = int(applicable.sum())
#         except Exception:
#             support = 0

#         setattr(c, "support_n", support)
#         setattr(c, "support", support)

#         # -----------------------------
#         # NEW: If inequality is always tight on its support, convert to equality
#         # touch == support means slack ≈ 0 for every applicable row.
#         # Only safe for non-strict inequalities Le/Ge.
#         # -----------------------------
#         if support > 0 and val == support:
#             R = getattr(c, "relation", None)
#             if isinstance(R, (Le, Ge)):
#                 left = R.left
#                 right = R.right

#                 # Canonicalize equality orientation to reduce duplicates:
#                 # choose a deterministic order based on repr
#                 try:
#                     if repr(right) < repr(left):
#                         left, right = right, left
#                 except Exception:
#                     pass

#                 # For dataset-derived equalities, tol=0.0 is usually what you want.
#                 # (You can use 1e-12 if you prefer numerical robustness.)
#                 c.relation = Eq(left, right, tol=0.0)

#         # -----------------------------
#         # Deduplicate AFTER possible rewrite
#         # -----------------------------
#         sig = c.signature()
#         if sig in seen:
#             continue
#         seen.add(sig)

#         unique.append(c)

#     unique.sort(
#         key=lambda cc: (
#             int(getattr(cc, "touch_count", 0)),
#             int(getattr(cc, "support_n", 0)),
#         ),
#         reverse=True,
#     )
#     return unique

def _annotate_and_sort_conjectures(
    df: pd.DataFrame,
    conjs: Sequence[Conjecture],
) -> List[Conjecture]:
    """
    Compute touch_count and support_n for each conjecture, normalize relations so
    logically equivalent forms deduplicate (e.g. L ≤ R vs R ≥ L), optionally
    upgrade non-strict inequalities to Eq when touch == support, then deduplicate
    by signature and sort.
    """
    from functools import reduce
    from math import gcd

    from txgraffiti.graffiti3.relations import Le, Lt, Ge, Gt, Eq

    def _canon_relation_inplace(c: Conjecture) -> None:
        """
        Canonicalize the conjecture relation so that:
          - Ge(L,R) becomes Le(R,L)
          - Gt(L,R) becomes Lt(R,L)
          - Eq(L,R) is ordered deterministically
        This makes equivalent conjectures share the same signature.
        """
        R = getattr(c, "relation", None)
        if R is None:
            return

        # Prefer only ≤, <, = in canonical form
        if isinstance(R, Ge):
            c.relation = Le(R.right, R.left)
            R = c.relation
        elif isinstance(R, Gt):
            c.relation = Lt(R.right, R.left)
            R = c.relation

        # Canonicalize equality orientation
        if isinstance(R, Eq):
            L, RR = R.left, R.right
            try:
                if repr(RR) < repr(L):
                    c.relation = Eq(RR, L, tol=getattr(R, "tol", 0.0))
            except Exception:
                pass

    unique: List[Conjecture] = []
    seen: set[str] = set()

    for c in conjs:
        # -----------------------------
        # 1) Compute touch_count
        # -----------------------------
        touch_attr = getattr(c, "touch_count", None)
        if callable(touch_attr):
            try:
                val = c.touch_count(df, auto_base=False)
            except TypeError:
                val = c.touch_count(df)
            except Exception:
                val = 0
        else:
            val = touch_attr if isinstance(touch_attr, int) else 0

        val = int(val)
        setattr(c, "touch_count", val)
        setattr(c, "touch", val)  # backward compatibility

        # -----------------------------
        # 2) Compute support_n
        # -----------------------------
        try:
            applicable, _, _ = c.check(df, auto_base=False)
            support = int(applicable.sum())
        except Exception:
            support = 0

        setattr(c, "support_n", support)
        setattr(c, "support", support)

        # -----------------------------
        # 3) If inequality is always tight on its support, upgrade to equality
        #    (touch == support) only meaningful for ≤ or ≥
        # -----------------------------
        if support > 0 and val == support:
            R = getattr(c, "relation", None)
            if isinstance(R, (Le, Ge)):
                left = R.left
                right = R.right
                # Canonicalize equality orientation
                try:
                    if repr(right) < repr(left):
                        left, right = right, left
                except Exception:
                    pass
                c.relation = Eq(left, right, tol=0.0)

        # -----------------------------
        # 4) Canonicalize relation so ≥/ > duplicates collapse into ≤/ <
        # -----------------------------
        _canon_relation_inplace(c)

        # -----------------------------
        # 5) Deduplicate AFTER canonicalization
        # -----------------------------
        sig = c.signature()
        if sig in seen:
            continue
        seen.add(sig)

        unique.append(c)

    # -----------------------------
    # 6) Sort
    # -----------------------------
    unique.sort(
        key=lambda cc: (
            int(getattr(cc, "touch_count", 0)),
            int(getattr(cc, "support_n", 0)),
        ),
        reverse=True,
    )
    return unique
