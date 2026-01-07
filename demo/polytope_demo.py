# In the terminal run the command: PYTHONPATH=src python demo/polytope_demo.py
from __future__ import annotations

import pandas as pd
import numpy as np

# at the top of graffiti4.py
from txgraffiti.graffiti3.heuristics.morgan import morgan_filter#, dalmatian_filter
from txgraffiti.graffiti3.heuristics.dalmatian import dalmatian_filter
from txgraffiti.graffiti3.graffiti3 import Graffiti3, Stage
from txgraffiti.example_data import polytope_data as df

# --- Start from your existing dataframe -------------------------------------

# Optional: keep a "clean" base view without the temperature columns
base_cols = [
    'simple_polytope',
    'p3', 'p4', 'p5', 'p6', 'p7', 'sum(pk_for_k>7)',
    'n', 'independence_number'
]
df_poly = df[base_cols].copy()

# Rename sum(pk_for_k>7) to something readable
df_poly = df_poly.rename(columns={'sum(pk_for_k>7)': 'p8plus'})

# --- 1. Simple face-count aggregations --------------------------------------

df_poly['N_small']   = df_poly['p3'] + df_poly['p4'] + df_poly['p5']
df_poly['N_big']     = df_poly['p7'] + df_poly['p8plus']       # all k ≥ 7
df_poly['N_nonhex']  = df_poly['N_small'] + df_poly['N_big']

# "Fullerene defect": faces that are NOT 5- or 6-gons
df_poly['fullerene_defect_count'] = (
    df_poly['p3'] + df_poly['p4'] + df_poly['p7'] + df_poly['p8plus']
)

# Total number of facets (this uses p6, so treat as "tainted" when targeting p6)
df_poly['f2_facets'] = (
    df_poly['p3'] + df_poly['p4'] + df_poly['p5']
    + df_poly['p6'] + df_poly['p7'] + df_poly['p8plus']
)

# --- 2. Curvature-style invariants (no p6 on RHS) ---------------------------
# Sum_{k<6} (6-k) p_k = 3 p3 + 2 p4 + p5

df_poly['C_small']      = 3*df_poly['p3'] + 2*df_poly['p4'] + df_poly['p5']
df_poly['big_curv_sum'] = df_poly['C_small'] - 12  # = Σ_{k≥7}(k-6) p_k

# Extra curvature beyond 7-gons (Σ_{k≥8}(k-7)p_k)
df_poly['extra_curv_8plus'] = df_poly['big_curv_sum'] - df_poly['N_big']

# Normalized versions (handle division by zero nicely)
df_poly['curv_per_small_face'] = np.where(
    df_poly['N_small'] > 0,
    df_poly['C_small'] / df_poly['N_small'],
    np.nan
)

df_poly['curv_per_big_face'] = np.where(
    df_poly['N_big'] > 0,
    df_poly['big_curv_sum'] / df_poly['N_big'],
    np.nan
)

# --- 3. Independence / vertex-based invariants ------------------------------

df_poly['alpha']          = df_poly['independence_number']
df_poly['alpha_ratio']    = df_poly['alpha'] / df_poly['n']          # α / n
df_poly['alpha_half_gap'] = (df_poly['n'] / 2.0) - df_poly['alpha']  # (n/2) - α

# Heuristic mix of independence vs small faces
df_poly['alpha_small_mix'] = df_poly['alpha'] / (df_poly['N_small'] + 1.0)

# --- 4. Curvature residue R_small (greedy, cheap–faces-first) --------------
def curvature_residue_small(row):
    """
    Greedy 'annihilation-style' curvature residue:
    budget B = 12 units, small faces cost:
      pentagon: 1, square: 2, triangle: 3
    Use as many cheap faces as possible: 5-gons, then 4-gons, then 3-gons.
    Return the max number of small faces we can fit in the budget.
    """
    B  = 12
    p3 = int(row['p3'])
    p4 = int(row['p4'])
    p5 = int(row['p5'])

    # Use pentagons first (cost 1)
    a5 = min(p5, B)
    B -= a5

    # Then squares (cost 2)
    a4 = min(p4, B // 2)
    B -= 2 * a4

    # Then triangles (cost 3)
    a3 = min(p3, B // 3)
    B -= 3 * a3

    return a3 + a4 + a5

df_poly['R_small'] = df_poly.apply(curvature_residue_small, axis=1)

# Optional: also define the "heavy-first" variant (triangles, then squares, then pentagons)
def curvature_residue_small_heavy(row):
    B  = 12
    p3 = int(row['p3'])
    p4 = int(row['p4'])
    p5 = int(row['p5'])

    # Triangles first
    a3 = min(p3, B // 3)
    B -= 3 * a3

    # Squares next
    a4 = min(p4, B // 2)
    B -= 2 * a4

    # Pentagons last
    a5 = min(p5, B)
    B -= a5

    return a3 + a4 + a5

df_poly['R_small_heavy'] = df_poly.apply(curvature_residue_small_heavy, axis=1)

# --- 5. Boolean predicates for Sophie / hypothesis lattice -----------------

# "Fullerene-like": only 5- and 6-gons and exactly 12 pentagons
df_poly['is_fullerene_like'] = (
    (df_poly['p3'] == 0) &
    (df_poly['p4'] == 0) &
    (df_poly['p7'] == 0) &
    (df_poly['p8plus'] == 0) &
    (df_poly['p5'] == 12)
)

df_poly['has_big_faces']      = df_poly['N_big'] > 0
df_poly['all_big_are_7gons']  = (df_poly['N_big'] > 0) & (df_poly['extra_curv_8plus'] == 0)
df_poly['more_triangles_than_squares'] = df_poly['p3'] > df_poly['p4']

# "Many small faces": threshold by, say, upper quartile of N_small
small_face_threshold = df_poly['N_small'].quantile(0.75)
df_poly['many_small_faces'] = df_poly['N_small'] >= small_face_threshold

# At this point df_poly has all the new columns and is ready for Graffiti3

g3 = Graffiti3(
    df_poly, # input dataframe
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
    # Stage.EXP_EXPONENT,
]

# Target invariants to conjecture on: p5 and p6.
TARGETS = [
        "p6",
        "p8plus",
        "independence_number",
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
    show_k_conjectures=15,
)
