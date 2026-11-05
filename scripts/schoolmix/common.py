"""Shared pieces for the school-mixture estimates of CAASPP grade-11 math
score distributions by group.

Inputs
------
- data/published_caaspp_grade11_by_group.csv (this repository)
- data/published_caaspp_grade11_score_histogram.csv (this repository)
- data/school_levels_grade11_math.csv and data/student_groups.csv from
  https://github.com/Hart-Hornor-Jones/ca-sbac-math-levels, passed by path.

Every file is opened with encoding='utf-8'.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(REPO, "data")

FLOOR = 2280
CUTS = (2543, 2628, 2718)          # first score of Levels 2, 3 and 4, grade 11 math
CEILING = {2015: 2862, 2016: 2862, 2017: 2862, 2019: 2862,
           2021: 2900, 2022: 2900, 2023: 2900, 2024: 2900, 2025: 2900}
TOP_CUTS = (1.0, 5.0, 10.0, 25.0)  # 'top X% of takers' cuts used in the comparison

# school-file group codes -> (output group_type, output group name) as used in
# data/estimated_share_above_cut.csv.  Filipino (77) is folded into Asian for
# output, as the existing estimates do, but is fitted as its own published group.
RACE_CODES = {75: "American Indian", 76: "Asian", 77: "Filipino", 74: "Black",
              78: "Hispanic/Latino", 79: "Pacific Islander", 80: "White", 144: "Two or more"}
ECON_CODES = {31: "Economically disadvantaged", 111: "Not economically disadvantaged"}
OUTPUT_RACE = {"American Indian": [75], "Asian": [76, 77], "Black": [74], "Hispanic/Latino": [78],
               "Pacific Islander": [79], "White": [80], "Two or more": [144]}
OUTPUT_ECON = {"Economically disadvantaged": [31], "Not economically disadvantaged": [111]}

# published names in data/published_caaspp_grade11_by_group.csv -> school-file code
PUB_TO_CODE = {"All students": 1, "Disadvantaged": 31, "Not disadvantaged": 111,
               "Black or African American": 74, "American Indian or Alaska Native": 75,
               "Asian": 76, "Filipino": 77, "Hispanic or Latino": 78,
               "Native Hawaiian or Pacific Islander": 79, "White": 80, "Two or more races": 144}


def school_year(year: int) -> str:
    return f"{year - 1}-{str(year)[2:]}"


def load_published():
    g = pd.read_csv(os.path.join(DATA, "published_caaspp_grade11_by_group.csv"), encoding="utf-8")
    g = g[g.subject == "Math"].copy()
    g["year"] = g.school_year.str[:4].astype(int) + 1
    g["code"] = g.group.map(PUB_TO_CODE)
    g = g[g.group_type.isin(["All", "Race/Ethnicity", "Economic status"])]
    h = pd.read_csv(os.path.join(DATA, "published_caaspp_grade11_score_histogram.csv"), encoding="utf-8")
    h = h[h.subject == "Math"].copy()
    h["year"] = h.school_year.str[:4].astype(int) + 1
    return g, h


def load_school(school_csv: str) -> pd.DataFrame:
    s = pd.read_csv(school_csv, dtype={"cds": str}, encoding="utf-8")
    keep = [1] + list(RACE_CODES) + list(ECON_CODES)
    s = s[s.group_code.isin(keep)].copy()
    # Five rows in the source file carry students_with_scores with '100' glued
    # on (16 -> 16100).  Cap at students_tested, which cannot be exceeded.
    bad = s.students_with_scores > s.students_tested
    s.loc[bad, "students_with_scores"] = s.loc[bad, "students_tested"]
    s["known"] = s.mean_scale_score.notna() & (s.students_with_scores > 0)
    return s


def published_row(g: pd.DataFrame, year: int, code: int) -> pd.Series:
    r = g[(g.year == year) & (g.code == code)]
    assert len(r) == 1, (year, code)
    return r.iloc[0]


def histogram(h: pd.DataFrame, year: int) -> pd.DataFrame:
    hy = h[h.year == year].sort_values("score_from").reset_index(drop=True)
    return hy[["score_from", "score_to", "n_students"]]


# ---------------------------------------------------------------- cells
def fit_cell_sd(mu: np.ndarray, c1: np.ndarray, c2: np.ndarray, c3: np.ndarray,
                sd_grid=np.arange(10.0, 400.5, 1.0)):
    """Least-squares SD of a normal with mean mu whose shares at or above the
    three cuts best match c1, c2, c3 (fractions).  Vectorized over cells."""
    mu = np.asarray(mu, float)[:, None]
    obs = np.stack([c1, c2, c3], axis=1).astype(float)
    pred = np.stack([norm.sf((cut - mu) / sd_grid[None, :]) for cut in CUTS], axis=2)  # cells x sd x 3
    err = ((pred - obs[:, None, :]) ** 2).sum(axis=2)
    best = err.argmin(axis=1)
    return sd_grid[best], err[np.arange(len(best)), best]


def prepare_cells(s: pd.DataFrame, year: int, codes) -> pd.DataFrame:
    """Known (unsuppressed) cells for the given group codes in a year, with the
    fitted within-cell SD and a 'degenerate' flag for cells whose level shares
    sit entirely in one level (then the SD takes the group's weighted median)."""
    d = s[(s.year == year) & s.known & s.group_code.isin(list(codes))].copy()
    p = d[["pct_level1", "pct_level2", "pct_level3", "pct_level4"]].to_numpy(float) / 100.0
    p = np.clip(p, 0, None)
    p = p / p.sum(axis=1, keepdims=True)          # levels add to 99-102 before this
    c1 = p[:, 1:].sum(axis=1); c2 = p[:, 2:].sum(axis=1); c3 = p[:, 3]
    sd, err = fit_cell_sd(d.mean_scale_score.to_numpy(), c1, c2, c3)
    degenerate = (p.max(axis=1) >= 0.999)
    d["sd_fit"] = sd
    d["fit_err"] = err
    d["degenerate"] = degenerate
    for k in range(4):
        d[f"p{k+1}"] = p[:, k]
    if degenerate.any():
        good = d[~d.degenerate]
        if len(good):
            order = np.argsort(good.sd_fit.to_numpy())
            w = good.students_with_scores.to_numpy()[order]
            med = good.sd_fit.to_numpy()[order][np.searchsorted(np.cumsum(w), w.sum() / 2)]
        else:
            med = 100.0
        d.loc[d.degenerate, "sd_fit"] = med
    return d


def normal_on_grid(mu, sd, grid, ceiling):
    """Discretized normal on integer scores FLOOR..ceiling: mass below FLOOR
    piles at FLOOR, mass at or above the ceiling piles at the ceiling."""
    edges = np.concatenate([[-np.inf], grid[:-1] + 0.5, [np.inf]])
    cdf = norm.cdf((edges - mu) / sd)
    pm = np.diff(cdf)
    # ceiling: everything from ceiling-0.5 up sits at the ceiling (already the last bin)
    return pm


def level_index(grid):
    lv = np.zeros(len(grid), int)
    for c in CUTS:
        lv += (grid >= c)
    return lv


def cell_distributions(cells: pd.DataFrame, grid: np.ndarray, ceiling: int) -> np.ndarray:
    """Level-anchored normal for every cell: within each achievement level the
    shape is the cell's fitted normal, and the mass in each level equals the
    cell's published level share.  Returns cells x grid."""
    lv = level_index(grid)
    out = np.zeros((len(cells), len(grid)))
    mus = cells.mean_scale_score.to_numpy(float)
    sds = cells.sd_fit.to_numpy(float)
    P = cells[["p1", "p2", "p3", "p4"]].to_numpy(float)
    for i in range(len(cells)):
        pm = normal_on_grid(mus[i], sds[i], grid, ceiling)
        for k in range(4):
            m = lv == k
            tot = pm[m].sum()
            if tot > 1e-300:
                out[i, m] = pm[m] / tot * P[i, k]
            elif P[i, k] > 0:
                # the fitted normal has essentially no mass in this level: put the
                # published share at the level edge nearest the mean
                idx = np.where(m)[0]
                j = idx[0] if mus[i] < grid[idx[0]] else idx[-1]
                out[i, j] = P[i, k]
    return out


def mixture(cells: pd.DataFrame, grid, ceiling):
    D = cell_distributions(cells, grid, ceiling)
    w = cells.students_with_scores.to_numpy(float)
    return (w[:, None] * D).sum(axis=0) / w.sum(), D


def share_at_or_above(f: np.ndarray) -> np.ndarray:
    return np.cumsum(f[::-1])[::-1]


# ---------------------------------------------------------------- LP grid
def lp_bins(year: int, extra_edges=(), coarse_low=False):
    """Bins for the linear programs: the published 10-point histogram bins,
    split at the three level cuts, at any extra cut scores, and with the
    ceiling as its own one-point bin.  Returns DataFrame lo, hi (inclusive),
    mid, width, hist_bin (index of the published histogram row)."""
    ceiling = CEILING[year]
    edges = set(range(FLOOR, ceiling + 1, 10)) | set(CUTS) | set(int(e) for e in extra_edges) | {ceiling, ceiling + 1}
    if coarse_low:   # 30-point bins below Level 2
        edges = {e for e in edges if e >= CUTS[0] or (e - FLOOR) % 30 == 0}
    edges = sorted(e for e in edges if FLOOR <= e <= ceiling + 1)
    rows = []
    for lo, nxt in zip(edges[:-1], edges[1:]):
        hi = nxt - 1
        rows.append(dict(lo=lo, hi=hi, mid=(lo + hi) / 2.0, width=hi - lo + 1))
    b = pd.DataFrame(rows)
    b["hist_bin"] = ((b.lo - FLOOR) // 10).astype(int)
    return b
