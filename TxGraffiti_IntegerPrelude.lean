import Mathlib

open scoped BigOperators

/-
TxGraffiti integer dataset prelude.

We provide (simple) Lean definitions matching the column names used
by the dataframe / lean_label mapping.
-/

-- Basic predicates
def prime (n : ℕ) : Prop := Nat.Prime n
def squarefree (n : ℕ) : Prop := Squarefree n

-- “Domain” predicate used as a base condition (optional)
def n_ge_2 (n : ℕ) : Prop := (2 : ℕ) ≤ n

-- Multiplicative invariants (Nat-valued)
noncomputable def omega (n : ℕ) : ℕ :=
  (Nat.primeFactors n).card

noncomputable def Omega (n : ℕ) : ℕ :=
  (Nat.factorization n).sum (fun _p e => e)

noncomputable def rad (n : ℕ) : ℕ :=
  (Nat.primeFactors n).prod id

noncomputable def phi (n : ℕ) : ℕ :=
  Nat.totient n

noncomputable def tau (n : ℕ) : ℕ :=
  (Nat.divisors n).card

noncomputable def sigma (n : ℕ) : ℕ :=
  ∑ d in (Nat.divisors n), d

-- Largest prime factor (returns 1 when n has no prime factors)
noncomputable def lpf (n : ℕ) : ℕ :=
  if h : (Nat.primeFactors n).Nonempty then (Nat.primeFactors n).max' h else 1

-- Carmichael function (λ), Möbius (μ), Liouville (λ_L)
-- If you don’t use these columns in Lean output, you can delete these defs.
noncomputable def lambda_carmichael (n : ℕ) : ℕ :=
  (ArithmeticFunction.Carmichael n)

noncomputable def mobius_mu (n : ℕ) : ℤ :=
  (ArithmeticFunction.moebius n)

noncomputable def liouville_lambda (n : ℕ) : ℤ :=
  (ArithmeticFunction.liouville n)

-- Extra boolean columns used in the demo
def almost_squarefree (n : ℕ) : Prop :=
  (Omega n - omega n = 1) ∧ ¬ squarefree n

def smooth7 (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p ∣ n → p ≤ 7

-- Dataset-specific predicate (declare if you use it)
constant twin_friend : ℕ → Prop
