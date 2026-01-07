# demo/lean_matrix_demo_mathlib_friendly.py
from __future__ import annotations

import numpy as np
import pandas as pd

from txgraffiti.graffiti3.heuristics.morgan import morgan_filter
from txgraffiti.graffiti3.heuristics.dalmatian import dalmatian_filter
from txgraffiti.graffiti3.graffiti3 import Graffiti3, Stage

from pathlib import Path


def make_matrix_features_lean(d: int = 3, namespace: str = "TxGraffiti") -> str:
    """
    Generate a Lean shim file defining the symbols used by lean_label.
    Fixed to Mat := Matrix (Fin d) (Fin d) ℝ.

    Assumes mathlib imports.
    """
    return f"""\
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.LinearAlgebra.Matrix.Determinant
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Data.Real.Basic

open scoped BigOperators
open BigOperators

namespace {namespace}

abbrev Mat := Matrix (Fin {d}) (Fin {d}) ℝ

def mat_trace (A : Mat) : ℝ := Matrix.trace A
def mat_det (A : Mat) : ℝ := Matrix.det A
def mat_abs_det (A : Mat) : ℝ := |Matrix.det A|

def mat_diag_sum_abs (A : Mat) : ℝ :=
  ∑ i : Fin {d}, |A i i|

def mat_offdiag_sum_abs (A : Mat) : ℝ :=
  ∑ i : Fin {d}, ∑ j : Fin {d}, (if h : i = j then 0 else |A i j|)

def mat_entry_sum_abs (A : Mat) : ℝ :=
  ∑ i : Fin {d}, ∑ j : Fin {d}, |A i j|

-- Frobenius-squared and Frobenius norm (as Real.sqrt of the sum of squares)
def mat_frob_sq (A : Mat) : ℝ :=
  ∑ i : Fin {d}, ∑ j : Fin {d}, (A i j) ^ (2 : ℕ)

def mat_frob (A : Mat) : ℝ :=
  Real.sqrt (mat_frob_sq A)

-- A simple max-abs-entry feature (uses Finset.sup)
def mat_max_entry_abs (A : Mat) : ℝ :=
  Finset.univ.sup (fun i : Fin {d} =>
    Finset.univ.sup (fun j : Fin {d} => |A i j|))

-- Predicates (Prop)
def is_symmetric (A : Mat) : Prop := Aᵀ = A

def is_diagonal (A : Mat) : Prop :=
  ∀ i j : Fin {d}, i ≠ j → A i j = 0

def is_upper_triangular (A : Mat) : Prop :=
  ∀ i j : Fin {d}, i > j → A i j = 0

def trace_zero (A : Mat) : Prop := mat_trace A = 0
def det_zero (A : Mat) : Prop := mat_det A = 0
def is_invertible (A : Mat) : Prop := mat_det A ≠ 0

def is_idempotent (A : Mat) : Prop := A ⬝ A = A
def is_involutory (A : Mat) : Prop := A ⬝ A = 1
def is_nilpotent2 (A : Mat) : Prop := A ⬝ A = 0

end {namespace}
"""


def write_and_print_lean_shim(*, out_path: str, d: int = 3, namespace: str = "TxGraffiti") -> str:
    """
    Write MatrixFeatures.lean and also return the text so the caller can print it.
    """
    text = make_matrix_features_lean(d=d, namespace=namespace)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return text

def build_matrix_dataframe_mathlib(
    N: int = 1500,
    d: int = 3,
    entry_min: int = -2,
    entry_max: int = 2,
    seed: int = 0,
) -> pd.DataFrame:
    """
    A matrix dataset designed so that every column corresponds to something
    you can define cleanly in Lean+mathlib.

    Numeric columns (ℝ):
      - tr          = trace
      - det         = det
      - abs_det     = |det|
      - diag_sum_abs = Σ_i |A_ii|
      - offdiag_sum_abs = Σ_{i≠j} |A_ij|
      - entry_sum_abs = Σ_{i,j} |A_ij|
      - max_entry_abs = max_{i,j} |A_ij|
      - frob_sq     = Σ_{i,j} (A_ij)^2
      - frob        = sqrt(frob_sq)

    Boolean columns:
      - is_symmetric
      - is_diagonal
      - is_upper_triangular
      - trace_zero
      - det_zero
      - is_invertible   (det ≠ 0)
      - is_idempotent   (A*A = A)
      - is_involutory   (A*A = I)
      - is_nilpotent2   (A*A = 0)
    """
    assert d == 3, "For now keep d=3 so Lean binder Matrix (Fin 3) (Fin 3) ℝ matches."

    rng = np.random.default_rng(seed)
    mats = rng.integers(entry_min, entry_max + 1, size=(N, d, d), dtype=np.int64)

    # avoid too many all-zero matrices
    for i in range(N):
        if np.all(mats[i] == 0):
            mats[i, 0, 0] = 1

    I = np.eye(d, dtype=np.int64)
    Z = np.zeros((d, d), dtype=np.int64)

    rows = []
    for A_int in mats:
        A = A_int.astype(float)

        tr = float(np.trace(A))
        det = float(np.linalg.det(A))
        abs_det = float(abs(det))

        diag = np.diag(A)
        diag_sum_abs = float(np.sum(np.abs(diag)))

        offdiag = A - np.diag(diag)
        offdiag_sum_abs = float(np.sum(np.abs(offdiag)))

        entry_sum_abs = float(np.sum(np.abs(A)))
        max_entry_abs = float(np.max(np.abs(A)))

        frob_sq = float(np.sum(A * A))
        frob = float(np.sqrt(frob_sq))

        # Boolean features (exact on integer matrices)
        is_sym = bool(np.array_equal(A_int, A_int.T))
        is_diag = bool(np.all(A_int == np.diag(np.diag(A_int))))
        is_upper = bool(np.all(np.tril(A_int, k=-1) == 0))

        trace_zero = (int(np.trace(A_int)) == 0)
        det_zero = (abs(det) <= 1e-9)
        is_invertible = not det_zero

        AA = A_int @ A_int
        is_idempotent = bool(np.array_equal(AA, A_int))
        is_involutory = bool(np.array_equal(AA, I))
        is_nilpotent2 = bool(np.array_equal(AA, Z))

        rows.append(
            dict(
                tr=tr,
                det=det,
                abs_det=abs_det,
                diag_sum_abs=diag_sum_abs,
                offdiag_sum_abs=offdiag_sum_abs,
                entry_sum_abs=entry_sum_abs,
                max_entry_abs=max_entry_abs,
                frob_sq=frob_sq,
                frob=frob,

                is_symmetric=is_sym,
                is_diagonal=is_diag,
                is_upper_triangular=is_upper,
                trace_zero=trace_zero,
                det_zero=det_zero,
                is_invertible=is_invertible,
                is_idempotent=is_idempotent,
                is_involutory=is_involutory,
                is_nilpotent2=is_nilpotent2,
            )
        )

    df = pd.DataFrame(rows)

    # Ensure booleans are really bool dtype
    for c in [
        "is_symmetric", "is_diagonal", "is_upper_triangular",
        "trace_zero", "det_zero", "is_invertible",
        "is_idempotent", "is_involutory", "is_nilpotent2",
    ]:
        df[c] = df[c].astype(bool)

    return df


if __name__ == "__main__":
    df = build_matrix_dataframe_mathlib(N=1500, d=3, entry_min=-2, entry_max=2, seed=0)

    # Lean label mapping: these names will be defined in a Lean shim file (see below)
    lean_label = {
        "__binders__": [("A", "Matrix (Fin 3) (Fin 3) ℝ")],
        "__defaults__": {"num_type": "ℝ"},

        # numeric columns (all ℝ, to keep Lean emission simple)
        "tr": {"term": "mat_trace", "type": "ℝ"},
        "det": {"term": "mat_det", "type": "ℝ"},
        "abs_det": {"term": "mat_abs_det", "type": "ℝ"},
        "diag_sum_abs": {"term": "mat_diag_sum_abs", "type": "ℝ"},
        "offdiag_sum_abs": {"term": "mat_offdiag_sum_abs", "type": "ℝ"},
        "entry_sum_abs": {"term": "mat_entry_sum_abs", "type": "ℝ"},
        "max_entry_abs": {"term": "mat_max_entry_abs", "type": "ℝ"},
        "frob_sq": {"term": "mat_frob_sq", "type": "ℝ"},
        "frob": {"term": "mat_frob", "type": "ℝ"},

        # boolean columns -> predicates (Prop)
        "is_symmetric": {"term": "is_symmetric", "type": "Prop", "kind": "pred"},
        "is_diagonal": {"term": "is_diagonal", "type": "Prop", "kind": "pred"},
        "is_upper_triangular": {"term": "is_upper_triangular", "type": "Prop", "kind": "pred"},
        "trace_zero": {"term": "trace_zero", "type": "Prop", "kind": "pred"},
        "det_zero": {"term": "det_zero", "type": "Prop", "kind": "pred"},
        "is_invertible": {"term": "is_invertible", "type": "Prop", "kind": "pred"},
        "is_idempotent": {"term": "is_idempotent", "type": "Prop", "kind": "pred"},
        "is_involutory": {"term": "is_involutory", "type": "Prop", "kind": "pred"},
        "is_nilpotent2": {"term": "is_nilpotent2", "type": "Prop", "kind": "pred"},
    }

    g3 = Graffiti3(
        df,
        max_boolean_arity=2,
        morgan_filter=morgan_filter,
        dalmatian_filter=dalmatian_filter,
        sophie_cfg=dict(
            eq_tol=1e-6,
            min_target_support=10,
            min_h_support=5,
            max_violations=0,
            min_new_coverage=1,
        ),
        lean_label=lean_label,
    )

    STAGES = [
        Stage.CONSTANT,
        Stage.RATIO,
        # Stage.LP1,
        # Stage.LP2,
        # Stage.LP3,
        # Stage.LP4,
        # Stage.POLY_SINGLE,
        # Stage.SQRT,
    ]

    TARGETS = ["abs_det", "frob", "max_entry_abs", "diag_sum_abs"]

    result = g3.conjecture(
        targets=TARGETS,
        stages=STAGES,
        include_invariant_products=False,
        include_abs=False,
        include_min_max=True,   # min/max are fine over ℝ
        include_log=False,
        enable_sophie=True,
        sophie_stages=STAGES,
        quick=True,
        show=True,
        show_k_conjectures=20,
    )

        # 1) generate + write shim
    shim_text = write_and_print_lean_shim(
        out_path="demo/MatrixFeatures.lean",   # change to wherever your Lean project expects it
        d=3,
        namespace="TxGraffiti",
    )

    print("\n==================== Lean shim (auto-generated) ====================\n")
    print(shim_text)
    print("\n==================== Conjectures ====================\n")

    print("\n\n--- Necessary (Lean) ---\n")
    print("\n \n".join(g3.conjectures_as_lean(result.conjectures, prefix="MatrixNecessary", start_index=1)[:20]))

    print("\n\n--- Sophie (Lean) ---\n")
    print("\n \n".join(g3.sophie_conditions_as_lean(result.sophie_conditions, prefix="MatrixSufficient", start_index=1)[:20]))
