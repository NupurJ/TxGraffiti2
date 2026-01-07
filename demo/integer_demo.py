
# Demo of Graffiti3 on integer data.
import math
from typing import Dict, List, Tuple
import pandas as pd

# Graffiti3 imports – adjust if your package layout changed
from txgraffiti.graffiti3.graffiti3 import Graffiti3, Stage, print_g3_result
from txgraffiti.graffiti3.heuristics.morgan import morgan_filter
from txgraffiti.graffiti3.heuristics.dalmatian import dalmatian_filter


# ------------------------------------------------------------
#  Prime factorization and basic multiplicative invariants
# ------------------------------------------------------------

def prime_factorization(n: int) -> List[Tuple[int, int]]:
    """
    Return the prime factorization of n as a list of (p, e),
    with p prime and e >= 1, sorted by p.
    """
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
        d += 1 if d == 2 else 2  # after 2, check only odd
    if m > 1:
        factors.append((m, 1))
    return factors


def multiplicative_invariants(n: int) -> Dict[str, float]:
    """
    Compute classical multiplicative invariants of n:
    - omega (ω): number of distinct prime factors
    - Omega (Ω): number of prime factors with multiplicity
    - tau: number of divisors
    - sigma: sum of divisors
    - radical: product of distinct prime factors
    - phi: Euler totient
    - lambda_carmichael: Carmichael function λ(n)
    - mobius_mu: Möbius μ(n)
    - liouville_lambda: Liouville λ_L(n)
    """
    if n == 1:
        return {
            "omega": 0,
            "Omega": 0,
            "tau": 1,
            "sigma": 1,
            "radical": 1,
            "phi": 1,
            "lambda_carmichael": 1,
            "mobius_mu": 1,
            "liouville_lambda": 1,
        }

    factors = prime_factorization(n)

    omega = len(factors)
    Omega = sum(e for _, e in factors)
    tau = 1
    sigma = 1
    radical = 1
    phi = n

    # tau, sigma, radical, phi
    for p, e in factors:
        tau *= (e + 1)
        sigma *= (p ** (e + 1) - 1) // (p - 1)
        radical *= p
        phi = phi // p * (p - 1)

    # Möbius μ(n)
    if any(e > 1 for _, e in factors):
        mobius_mu = 0
    else:
        mobius_mu = -1 if omega % 2 == 1 else 1

    # Liouville λ_L(n)
    liouville_lambda = -1 if Omega % 2 == 1 else 1

    # Carmichael λ(n): lcm of λ(p^e)
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

    return {
        "omega": omega,
        "Omega": Omega,
        "tau": tau,
        "sigma": sigma,
        "radical": radical,
        "phi": phi,
        "lambda_carmichael": lambda_carmichael,
        "mobius_mu": mobius_mu,
        "liouville_lambda": liouville_lambda,
    }


# ------------------------------------------------------------
#  Creative numeric features
# ------------------------------------------------------------

def creative_numeric_features(n: int, inv: Dict[str, float]) -> Dict[str, float]:
    """
    Construct some 'creative' numeric columns that number theorists
    might find interesting as raw material:

    - log_n, loglog_n
    - n_over_phi = n / phi(n)
    - sigma_over_n = sigma(n) / n
    - normalized_sigma = sigma(n) / (n * log log n) (when defined)
    - totient_defect = n - phi(n)
    - avg_log_prime_factor
    - entropy_prime_exponents: Shannon entropy of exponent distribution
    - radical_over_n = radical(n) / n
    """
    n_float = float(n)
    log_n = math.log(n_float) if n > 1 else 0.0
    loglog_n = math.log(log_n) if log_n > 0 else float("nan")

    phi = inv["phi"]
    sigma = inv["sigma"]
    radical = inv["radical"]
    Omega = inv["Omega"]

    # n / phi(n)
    n_over_phi = n_float / phi if phi > 0 else float("inf")

    # sigma(n) / n
    sigma_over_n = sigma / n_float

    # sigma(n) / (n log log n)
    if n > 2 and loglog_n > 0:
        normalized_sigma = sigma / (n_float * loglog_n)
    else:
        normalized_sigma = float("nan")

    # totient defect
    totient_defect = n - phi

    # average log prime factor (with multiplicity)
    factors = prime_factorization(n) if n > 1 else []
    if Omega > 0:
        sum_log_primes = sum(e * math.log(p) for p, e in factors)
        avg_log_prime_factor = sum_log_primes / Omega
    else:
        avg_log_prime_factor = float("nan")

    # entropy of prime exponent distribution
    if Omega > 0:
        probs = [e / Omega for _, e in factors]
        entropy_prime_exponents = -sum(
            p * math.log(p) for p in probs if p > 0
        )
    else:
        entropy_prime_exponents = 0.0

    # radical / n
    radical_over_n = radical / n_float

    return {
        "log_n": log_n,
        "loglog_n": loglog_n,
        "n_over_phi": n_over_phi,
        "sigma_over_n": sigma_over_n,
        "normalized_sigma": normalized_sigma,
        "totient_defect": totient_defect,
        "avg_log_prime_factor": avg_log_prime_factor,
        "entropy_prime_exponents": entropy_prime_exponents,
        "radical_over_n": radical_over_n,
    }


# ------------------------------------------------------------
#  Boolean properties (<= 4) designed for interesting implications
# ------------------------------------------------------------

def boolean_properties(n: int, inv: Dict[str, float]) -> Dict[str, bool]:
    """
    Boolean columns (exactly 4) that are plausible targets
    for new implications / Sophie-style characterizations:

    - is_prime: classical but central.
    - is_squarefree: all prime exponents are 0 or 1.
    - is_almost_squarefree: Ω(n) - ω(n) = 1 (exactly one extra prime factor),
      i.e., 'one square over' squarefree.
    - is_7_smooth: all prime factors <= 7 (7-smooth numbers).

    These are simple, interpretable predicates that Graffiti3 can
    try to characterize in terms of other invariants.
    """
    if n <= 1:
        factors = []
    else:
        factors = prime_factorization(n)

    # primality
    is_prime = (n > 1) and len(factors) == 1 and factors[0][1] == 1

    # squarefree
    is_squarefree = all(e == 1 for _, e in factors) if n > 1 else False

    # 'almost' squarefree: exactly one extra exponent beyond squarefree
    omega = inv["omega"]
    Omega = inv["Omega"]
    is_almost_squarefree = (Omega - omega == 1) and not is_squarefree

    # 7-smooth numbers: all primes <= 7
    is_7_smooth = all(p <= 7 for p, _ in factors) if n > 1 else False

    return {
        "integer >= 2": True,
        "prime": is_prime,
        "squarefree": is_squarefree,
        "almost_squarefree": is_almost_squarefree,
        "7_smooth": is_7_smooth,
    }


# ------------------------------------------------------------
#  Putting it all together
# ------------------------------------------------------------

def build_integer_dataframe(n_min: int = 2, n_max: int = 2000) -> pd.DataFrame:
    """
    Build a pandas DataFrame of integer data for n in [n_min, n_max].

    Columns include:
      - n
      - classical multiplicative invariants (omega, Omega, tau, sigma, radical,
        phi, lambda_carmichael, mobius_mu, liouville_lambda)
      - creative numeric features (log_n, loglog_n, n_over_phi, sigma_over_n,
        normalized_sigma, totient_defect, avg_log_prime_factor,
        entropy_prime_exponents, radical_over_n)
      - Boolean properties: is_prime, is_squarefree, is_almost_squarefree,
        is_7_smooth
    """
    records = []

    for n in range(n_min, n_max + 1):
        inv = multiplicative_invariants(n)
        # creat = creative_numeric_features(n, inv)
        bools = boolean_properties(n, inv)

        row = {"n": n}
        row.update(inv)
        # row.update(creat)
        row.update(bools)
        records.append(row)

    df = pd.DataFrame.from_records(records)
    return df


df = build_integer_dataframe()

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
    'phi',
    'lambda_carmichael',
]

# Conjecture on the target invariants using the stages defined above.
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
