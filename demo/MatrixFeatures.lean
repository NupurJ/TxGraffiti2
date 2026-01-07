import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.LinearAlgebra.Matrix.Determinant
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Data.Real.Basic

open scoped BigOperators
open BigOperators

namespace TxGraffiti

abbrev Mat := Matrix (Fin 3) (Fin 3) ℝ

def mat_trace (A : Mat) : ℝ := Matrix.trace A
def mat_det (A : Mat) : ℝ := Matrix.det A
def mat_abs_det (A : Mat) : ℝ := |Matrix.det A|

def mat_diag_sum_abs (A : Mat) : ℝ :=
  ∑ i : Fin 3, |A i i|

def mat_offdiag_sum_abs (A : Mat) : ℝ :=
  ∑ i : Fin 3, ∑ j : Fin 3, (if h : i = j then 0 else |A i j|)

def mat_entry_sum_abs (A : Mat) : ℝ :=
  ∑ i : Fin 3, ∑ j : Fin 3, |A i j|

-- Frobenius-squared and Frobenius norm (as Real.sqrt of the sum of squares)
def mat_frob_sq (A : Mat) : ℝ :=
  ∑ i : Fin 3, ∑ j : Fin 3, (A i j) ^ (2 : ℕ)

def mat_frob (A : Mat) : ℝ :=
  Real.sqrt (mat_frob_sq A)

-- A simple max-abs-entry feature (uses Finset.sup)
def mat_max_entry_abs (A : Mat) : ℝ :=
  Finset.univ.sup (fun i : Fin 3 =>
    Finset.univ.sup (fun j : Fin 3 => |A i j|))

-- Predicates (Prop)
def is_symmetric (A : Mat) : Prop := Aᵀ = A

def is_diagonal (A : Mat) : Prop :=
  ∀ i j : Fin 3, i ≠ j → A i j = 0

def is_upper_triangular (A : Mat) : Prop :=
  ∀ i j : Fin 3, i > j → A i j = 0

def trace_zero (A : Mat) : Prop := mat_trace A = 0
def det_zero (A : Mat) : Prop := mat_det A = 0
def is_invertible (A : Mat) : Prop := mat_det A ≠ 0

def is_idempotent (A : Mat) : Prop := A ⬝ A = A
def is_involutory (A : Mat) : Prop := A ⬝ A = 1
def is_nilpotent2 (A : Mat) : Prop := A ⬝ A = 0

end TxGraffiti
