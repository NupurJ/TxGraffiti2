# In the terminal run:
# PYTHONPATH=src python demo/lean_graph_demo.py

from __future__ import annotations
import pandas as pd

from txgraffiti.graffiti3.heuristics.morgan import morgan_filter
from txgraffiti.graffiti3.heuristics.dalmatian import dalmatian_filter
from txgraffiti.graffiti3.graffiti3 import Graffiti3, Stage
from txgraffiti.example_data import graph_data as df

# ───────────────────────────── data prep ─────────────────────────────
df = df.copy()

# All graphs in our dataset have order >= 2
df["nontrivial"] = df["connected"]

# Drop columns you don’t want to be used as invariants/properties in conjecturing
df.drop(
    columns=[
        # "tree",
        "chordal",
        # "cubic",
        # "triameter",
        "clique_number",
        "vertex_cover_number",
        "cograph",
        "size",
    ],
    inplace=True,
    errors="ignore",
)

# ───────────────────────────── lean labels ─────────────────────────────

lean_label: dict = {
    "__object__": {"name": "G", "type": "SimpleGraph V"},
    # Use ℚ as the ambient numeric type when the expression involves fractions.
    "__defaults__": {"num_type": "ℕ"},
}

# Columns you know are ℕ-valued invariants (counts)
N_COLS = {
    "order",
    "size",
    "maximum_degree",
    "minimum_degree",
    "diameter",
    "radius",
    "clique_number",
    "chromatic_number",
    "independence_number",
    "vertex_cover_number",
    "matching_number",
    "triameter",
    "slater",
    "annihilation_number",
    "residue",
    "domination_number",
    "total_domination_number",
    "independent_domination_number",
    "min_maximal_matching_number",
    "zero_forcing_number",
}

# ℚ-valued invariants (rare; keep explicit)
Q_COLS = {"harmonic_index"}

# ℝ-valued invariants (spectral stuff)
R_COLS = {
    "spectral_radius",
    "largest_laplacian_eigenvalue",
    "second_largest_adjacency_eigenvalue",
}

for col in df.columns:
    # Predicates
    if pd.api.types.is_bool_dtype(df[col]):
        lean_label[col] = {"term": col, "type": "Prop"}

    # Invariants: force known types first
    if col in N_COLS:
        ty = "ℕ"
    elif col in Q_COLS:
        ty = "ℚ"
    elif col in R_COLS:
        ty = "ℝ"
    else:
        # Fallback: prefer ℕ if column is integer dtype, else ℚ (NOT ℝ)
        # ℚ is safer because your conjectures frequently introduce rational coefficients.
        ty = "ℕ" if pd.api.types.is_integer_dtype(df[col]) else "ℚ"

    lean_label[col] = {"term": col, "type": ty}

# ───────────────────────────── run graffiti3 ─────────────────────────────

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

# STAGES is a list of methods for producing inequalties
STAGES = [
    Stage.CONSTANT,        # TxGraffiti capability
    Stage.RATIO,           # TxGraffiti capability
    Stage.LP1,             # TxGraffiti capability
    Stage.LP2,             # TxGraffiti capability
    # Stage.LP3,           # TxGraffiti capability
    # Stage.LP4,           # TxGraffiti capability
    # Stage.POLY_SINGLE,   # New capability
    # Stage.MIXED,         # New capability
    # Stage.SQRT,          # New capability
    # Stage.LOG,           # New capability
    # Stage.SQRT_LOG,      # New capability
    # Stage.GEOM_MEAN,     # New capability
    # Stage.LOG_SUM,       # New capability
    # Stage.SQRT_PAIR,     # New capability
    # Stage.SQRT_SUM,      # New capability
    # Stage.EXP_EXPONENT,  # New capability
]

TARGETS = [
    # "independence_number",
    # "radius",
    # "domination_number",
    # "matching_number",
    # "harmonic_index",
    # "residue",
    "zero_forcing_number",
]

g3.conjecture(
    targets=TARGETS,
    stages=STAGES,
    include_invariant_products=False, # New: compute a*b on noncomparable columns a and b
    include_abs=False, # New: compute |a-b| on all noncomparable columns a and b
    include_min_max=False, # New: compute min(a, b) and max(a, b) on all noncomparable columns a and b
    include_log=False, # New: Compute log(a) on all columns with a > 1
    enable_sophie=True, # New: Sophie-style sufficient condition conjecturing
    sophie_stages=STAGES,
    quick=True,
    show=True,
    show_k_conjectures=30,
)

print()
print()
print()
print("==================== Lean Theorems ======================")
print("\n \n".join(g3.lean_necessary_statements[:30]))
print()
print()
print("\n \n".join(g3.lean_sufficient_statements[:30]))
