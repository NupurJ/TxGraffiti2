# demo/lean_integer_demo.py
from __future__ import annotations

import pathlib
from typing import Dict, List, Tuple

import pandas as pd

from txgraffiti.graffiti3.graffiti3 import Graffiti3, Stage
from txgraffiti.graffiti3.heuristics.morgan import morgan_filter
from txgraffiti.graffiti3.heuristics.dalmatian import dalmatian_filter


# ------------------------------------------------------------
#  Your integer dataset (as you provided, lightly trimmed)
# ------------------------------------------------------------

def prime_factorization(n: int) -> List[Tuple[int, int]]:
    factors = []
    d = 2
    m = n
    while d * d <= m:
        if m % d == 0:
            e = 0
            while m % d == 0:
                m //= d
                e += 1
            factors.append((d, e))
        d += 1 if d == 2 else 2
    if m > 1:
        factors.append((m, 1))
    return factors


def multiplicative_invariants(n: int) -> Dict[str, int]:
    if n == 1:
        return dict(
            omega=0, Omega=0, tau=1, sigma=1, radical=1,
            phi=1, lambda_carmichael=1, mobius_mu=1, liouville_lambda=1
        )

    factors = prime_factorization(n)
    omega = len(factors)
    Omega = sum(e for _, e in factors)

    tau = 1
    sigma = 1
    radical = 1
    phi = n

    for p, e in factors:
        tau *= (e + 1)
        sigma *= (p ** (e + 1) - 1) // (p - 1)
        radical *= p
        phi = phi // p * (p - 1)

    # mobius
    if any(e > 1 for _, e in factors):
        mobius_mu = 0
    else:
        mobius_mu = -1 if (omega % 2 == 1) else 1

    # liouville
    liouville_lambda = -1 if (Omega % 2 == 1) else 1

    # carmichael (simple version)
    import math
    lambdas = []
    for p, e in factors:
        if p == 2:
            if e == 1:
                lam_pe = 1
            elif e == 2:
                lam_pe = 2
            else:
                lam_pe = 2 ** (e - 2)
        else:
            lam_pe = p ** (e - 1) * (p - 1)
        lambdas.append(lam_pe)
    lambda_carmichael = lambdas[0]
    for lam in lambdas[1:]:
        lambda_carmichael = math.lcm(lambda_carmichael, lam)

    return dict(
        omega=omega,
        Omega=Omega,
        tau=tau,
        sigma=sigma,
        radical=radical,
        phi=phi,
        lambda_carmichael=lambda_carmichael,
        mobius_mu=mobius_mu,
        liouville_lambda=liouville_lambda,
    )


def boolean_properties(n: int, inv: Dict[str, int]) -> Dict[str, bool]:
    factors = prime_factorization(n) if n > 1 else []
    is_prime = (n > 1) and len(factors) == 1 and factors[0][1] == 1
    is_squarefree = all(e == 1 for _, e in factors) if n > 1 else False

    omega = inv["omega"]
    Omega = inv["Omega"]
    is_almost_squarefree = (Omega - omega == 1) and (not is_squarefree)

    is_7_smooth = all(p <= 7 for p, _ in factors) if n > 1 else False

    return {
        "n_ge_2": (n >= 2),
        "prime": is_prime,
        "squarefree": is_squarefree,
        "almost_squarefree": is_almost_squarefree,
        "smooth7": is_7_smooth,
        # optional dataset-specific:
        # "twin_friend": ...  (if you compute it)
    }


def build_integer_dataframe(n_min: int = 2, n_max: int = 2000) -> pd.DataFrame:
    rows = []
    for n in range(n_min, n_max + 1):
        inv = multiplicative_invariants(n)
        bools = boolean_properties(n, inv)
        row = {"n": n}
        row.update(inv)
        row.update(bools)
        rows.append(row)
    df = pd.DataFrame(rows)

    # ensure bool dtypes
    for c in ["n_ge_2", "prime", "squarefree", "almost_squarefree", "smooth7"]:
        if c in df.columns:
            df[c] = df[c].astype(bool)

    return df


# ------------------------------------------------------------
#  Lean prelude generator
# ------------------------------------------------------------

def make_integer_prelude_lean() -> str:
    # Uses:
    #  - Nat.primeFactors : Finset ℕ   :contentReference[oaicite:1]{index=1}
    #  - Nat.divisors : Finset ℕ       :contentReference[oaicite:2]{index=2}
    #  - Nat.factorization             :contentReference[oaicite:3]{index=3}
    #  - Nat.Prime                     :contentReference[oaicite:4]{index=4}
    return """\
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
"""


# ------------------------------------------------------------
#  Main demo
# ------------------------------------------------------------

if __name__ == "__main__":
    df = build_integer_dataframe(n_min=2, n_max=2000)

    # Lean label mapping: point each dataframe column at a Lean term/type.
    # IMPORTANT: for numeric columns, set kind=None (default), for boolean columns set kind="pred".
    lean_label = {
        "__binders__": [("n", "ℕ")],
        "__defaults__": {"num_type": "ℕ"},

        # numeric columns
        "omega": {"term": "omega", "type": "ℕ"},
        "Omega": {"term": "Omega", "type": "ℕ"},
        "tau": {"term": "tau", "type": "ℕ"},
        "sigma": {"term": "sigma", "type": "ℕ"},
        "radical": {"term": "rad", "type": "ℕ"},
        "phi": {"term": "phi", "type": "ℕ"},
        "lambda_carmichael": {"term": "lambda_carmichael", "type": "ℕ"},
        "mobius_mu": {"term": "mobius_mu", "type": "ℤ"},
        "liouville_lambda": {"term": "liouville_lambda", "type": "ℤ"},
        "n": {"term": "{obj}", "type": "ℕ"},

        # boolean columns (predicates)
        "n_ge_2": {"term": "n_ge_2", "type": "Prop", "kind": "pred"},
        "prime": {"term": "prime", "type": "Prop", "kind": "pred"},
        "squarefree": {"term": "squarefree", "type": "Prop", "kind": "pred"},
        "almost_squarefree": {"term": "almost_squarefree", "type": "Prop", "kind": "pred"},
        "smooth7": {"term": "smooth7", "type": "Prop", "kind": "pred"},
        # "twin_friend": {"term": "twin_friend", "type": "Prop", "kind": "pred"},
    }

    g3 = Graffiti3(
        df,
        max_boolean_arity=2,
        morgan_filter=morgan_filter,
        dalmatian_filter=dalmatian_filter,
        sophie_cfg=dict(
            eq_tol=1e-4,
            min_target_support=5,
            min_h_support=3,
            max_violations=0,
            min_new_coverage=1,
        ),
        lean_label=lean_label,
    )

    STAGES = [
        # Stage.CONSTANT,
        Stage.RATIO,
        Stage.LP1,
        Stage.LP2,
        Stage.LP3,
        Stage.LP4,
        Stage.POLY_SINGLE,
        Stage.MIXED,
        Stage.SQRT,
        Stage.LOG,
        Stage.SQRT_LOG,
        Stage.GEOM_MEAN,
        Stage.LOG_SUM,
        Stage.SQRT_PAIR,
        Stage.SQRT_SUM,
        Stage.EXP_EXPONENT,
    ]

    TARGETS = [
        "phi",
        "lambda_carmichael",
        ]

    result = g3.conjecture(
        targets=TARGETS,
        stages=STAGES,
        include_invariant_products=False,
        include_abs=False,
        include_min_max=False,
        include_log=False,
        enable_sophie=True,
        sophie_stages=STAGES,
        quick=True,
        show=True,
        show_k_conjectures=10,
    )

    print("\n==================== Lean Prelude (.lean) ====================\n")
    prelude = make_integer_prelude_lean()
    print(prelude)

    print("\n==================== Lean Theorems ======================\n")
    print("\n--- Necessary conditions (Lean) ---\n")
    print("\n\n".join(g3.conjectures_as_lean(result.conjectures, prefix="IntNecessary", start_index=1)[:20]))

    print("\n\n--- Sufficient (Sophie) conditions (Lean) ---\n")
    print("\n\n".join(g3.sophie_conditions_as_lean(result.sophie_conditions, prefix="IntSufficient", start_index=1)[:20]))
