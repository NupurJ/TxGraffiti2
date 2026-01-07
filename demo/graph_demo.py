# In the terminal run:
# PYTHONPATH=src python demo/graph_demo.py

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
        "chordal",
        "clique_number",
        "vertex_cover_number",
        "size",
        "cograph",
        "cubic",
    ],
    inplace=True,
    errors="ignore",
)

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
    "independence_number",
    "annihilation_number",
    # "harmonic_index",
]

result = g3.conjecture(
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
    show_k_conjectures=20,
)
