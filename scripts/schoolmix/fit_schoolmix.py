"""Point estimates: school-mixture seeds adjusted (minimum relative entropy)
to the published statewide figures and the all-student histogram.

Usage: python fit_schoolmix.py /path/to/school_levels_grade11_math.csv

Writes data/estimated_share_above_cut_schoolmix.csv and
data/estimated_diagnostics_schoolmix.csv (plus the seeds as .npz for the
bounds and comparison scripts, in data/schoolmix_cache/).
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd
import cvxpy as cp

from common import (DATA, FLOOR, CUTS, CEILING, RACE_CODES, ECON_CODES, OUTPUT_RACE, OUTPUT_ECON,
                    load_published, load_school, published_row, histogram, prepare_cells, mixture,
                    normal_on_grid, level_index, share_at_or_above)

ROUND_PCT = 0.5      # printed whole percents are read as +-0.5
MEAN_TOL = 0.5       # printed integer means
SD_TOL = 1.5         # as in METHOD.md section 8
CACHE = os.path.join(DATA, "schoolmix_cache")


def seeds_for(s, g, year, codes):
    """Seed distribution for each published group code: known-cell mixture plus
    a remainder for suppressed/omitted cells seeded with the group's published
    normal.  Returns dict code -> dict."""
    ceiling = CEILING[year]
    grid = np.arange(FLOOR, ceiling + 1)
    n_all_cells = s[(s.year == year) & s.known & (s.group_code == 1)].students_with_scores.sum()
    pub_all = published_row(g, year, 1)
    out = {}
    for code in codes:
        pub = published_row(g, year, code)
        cells = prepare_cells(s, year, [code])
        n_known = float(cells.students_with_scores.sum())
        n_target = pub.n_tested / pub_all.n_tested * n_all_cells
        n_rem = max(0.0, n_target - n_known)
        q_rem = normal_on_grid(pub.mean_scale_score, pub.sd_scale_score, grid, ceiling)
        if n_known > 0:
            q_known, _ = mixture(cells, grid, ceiling)
        else:
            q_known = q_rem.copy()
        q = (n_known * q_known + n_rem * q_rem) / (n_known + n_rem)
        out[code] = dict(pub=pub, cells=cells, n_known=n_known, n_target=n_target, n_rem=n_rem,
                         q_known=q_known, q_rem=q_rem, q=q, grid=grid, ceiling=ceiling)
    return out


def moments(f, grid):
    m = (f * grid).sum()
    return m, np.sqrt(max((f * grid ** 2).sum() - m * m, 0.0))


def levels(f, grid):
    lv = level_index(grid)
    return np.array([f[lv == k].sum() for k in range(4)])


def adjust(seeds, g, year, h, scheme_name, verbose=True):
    """Minimum-relative-entropy adjustment of all groups of one scheme jointly."""
    codes = list(seeds)
    grid = seeds[codes[0]]["grid"]
    S = len(grid)
    pub_all = published_row(g, year, 1)
    w = np.array([seeds[c]["pub"].n_tested for c in codes], float)
    w = w / w.sum()
    hy = histogram(h, year)
    hshare = hy.n_students.to_numpy(float) / hy.n_students.sum()
    # histogram bin membership matrix (bins x scores)
    H = np.zeros((len(hy), S))
    for i, r in hy.iterrows():
        H[i, (grid >= r.score_from) & (grid <= r.score_to)] = 1.0
    lv = level_index(grid)
    L = np.stack([(lv == k).astype(float) for k in range(4)])

    F = cp.Variable((len(codes), S), nonneg=True)
    Q = np.stack([np.maximum(seeds[c]["q"], 1e-12) for c in codes])
    obj = 0
    cons = [cp.sum(F, axis=1) == 1]
    for i, c in enumerate(codes):
        obj = obj + w[i] * cp.sum(cp.rel_entr(F[i], Q[i]))
        pub = seeds[c]["pub"]
        p = np.array([pub.pct_level1_not_met, pub.pct_level2_nearly_met, pub.pct_level3_met, pub.pct_level4_exceeded], float) / 100
        cons += [L @ F[i] >= p - ROUND_PCT / 100, L @ F[i] <= p + ROUND_PCT / 100]
        cons += [grid @ F[i] >= pub.mean_scale_score - MEAN_TOL, grid @ F[i] <= pub.mean_scale_score + MEAN_TOL]
        m2_lo = (pub.sd_scale_score - SD_TOL) ** 2 + (pub.mean_scale_score - MEAN_TOL) ** 2
        m2_hi = (pub.sd_scale_score + SD_TOL) ** 2 + (pub.mean_scale_score + MEAN_TOL) ** 2
        cons += [(grid ** 2) @ F[i] >= m2_lo, (grid ** 2) @ F[i] <= m2_hi]
    cons += [H @ (F.T @ w) == hshare]
    prob = cp.Problem(cp.Minimize(obj), cons)
    t0 = time.time()
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception as e:  # pragma: no cover
        print("CLARABEL failed:", e, "- trying SCS")
        prob.solve(solver=cp.SCS, eps=1e-7, max_iters=200000)
    if verbose:
        print(f"  {year} {scheme_name}: status {prob.status}, objective {prob.value:.5f} nats/student, {time.time() - t0:.1f}s")
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"{year} {scheme_name}: {prob.status}")
    Fv = np.maximum(F.value, 0)
    Fv = Fv / Fv.sum(axis=1, keepdims=True)
    return Fv, prob.status, prob.value


def output_grid(year):
    ceiling = CEILING[year]
    sc = list(range(FLOOR, ceiling + 1, 5))
    if sc[-1] != ceiling:
        sc.append(ceiling)
    return np.array(sc)


def main(school_csv):
    s = load_school(school_csv)
    g, h = load_published()
    years = sorted(set(g.year) & set(s.year) & set(CEILING))
    os.makedirs(CACHE, exist_ok=True)
    rows, diag = [], []
    for year in years:
        print(f"year {year}")
        grid = np.arange(FLOOR, CEILING[year] + 1)
        og = output_grid(year)
        idx = og - FLOOR
        for scheme_name, codes, outmap, gtype in [("race", list(RACE_CODES), OUTPUT_RACE, "Race/ethnicity"),
                                                    ("econ", list(ECON_CODES), OUTPUT_ECON, "Economic status")]:
            seeds = seeds_for(s, g, year, codes)
            Fv, status, obj = adjust(seeds, g, year, h, scheme_name)
            np.savez(os.path.join(CACHE, f"fit_{year}_{scheme_name}.npz"), codes=np.array(codes), F=Fv,
                     Q=np.stack([seeds[c]["q"] for c in codes]), grid=grid)
            pubN = {c: seeds[c]["pub"].n_tested for c in codes}
            for i, c in enumerate(codes):
                sd_ = seeds[c]
                q, f = sd_["q"], Fv[i]
                qm, qs = moments(q, grid); fm, fs = moments(f, grid)
                ql, fl = levels(q, grid), levels(f, grid)
                sa_q, sa_f = share_at_or_above(q), share_at_or_above(f)
                pub = sd_["pub"]
                d = dict(year=year, scheme=scheme_name, group_published=pub.group, n_published=int(pub.n_tested),
                         n_known_cells=int(sd_["n_known"]), n_target=round(sd_["n_target"]), n_remainder=round(sd_["n_rem"]),
                         coverage=round(sd_["n_known"] / sd_["n_target"], 4) if sd_["n_target"] > 0 else np.nan,
                         cells=len(sd_["cells"]), cells_degenerate=int(sd_["cells"].degenerate.sum()),
                         cell_sd_median=float(np.median(sd_["cells"].sd_fit)) if len(sd_["cells"]) else np.nan,
                         pub_mean=pub.mean_scale_score, seed_mean=round(qm, 2), final_mean=round(fm, 2),
                         pub_sd=pub.sd_scale_score, seed_sd=round(qs, 2), final_sd=round(fs, 2),
                         pub_L1=pub.pct_level1_not_met, seed_L1=round(100 * ql[0], 2), final_L1=round(100 * fl[0], 2),
                         pub_L2=pub.pct_level2_nearly_met, seed_L2=round(100 * ql[1], 2), final_L2=round(100 * fl[1], 2),
                         pub_L3=pub.pct_level3_met, seed_L3=round(100 * ql[2], 2), final_L3=round(100 * fl[2], 2),
                         pub_L4=pub.pct_level4_exceeded, seed_L4=round(100 * ql[3], 2), final_L4=round(100 * fl[3], 2),
                         kl_final_vs_seed=round(float(np.sum(np.where(f > 0, f * np.log(np.maximum(f, 1e-300) / np.maximum(q, 1e-300)), 0))), 5),
                         max_abs_change_share_at_or_above=round(float(np.max(np.abs(sa_f - sa_q))), 4),
                         change_at_2800=round(float(sa_f[2800 - FLOOR] - sa_q[2800 - FLOOR]), 4),
                         change_at_2860=round(float(sa_f[2860 - FLOOR] - sa_q[2860 - FLOOR]), 4),
                         solver_status=status)
                diag.append(d)
            # output groups
            for gname, members in outmap.items():
                ii = [codes.index(c) for c in members]
                wN = np.array([pubN[c] for c in members], float)
                f = (wN[:, None] * Fv[ii]).sum(axis=0) / wN.sum()
                q = (wN[:, None] * np.stack([seeds[c]["q"] for c in members])).sum(axis=0) / wN.sum()
                sa_f, sa_q = share_at_or_above(f), share_at_or_above(q)
                for j, sc in enumerate(og):
                    rows.append(("CAASPP Math", year, gtype, gname, int(wN.sum()), int(sc),
                                 round(float(sa_f[idx[j]]), 5), round(float(sa_q[idx[j]]), 5)))
            if scheme_name == "race":
                wN = np.array([pubN[c] for c in codes], float)
                f = (wN[:, None] * Fv).sum(axis=0) / wN.sum()
                q = (wN[:, None] * np.stack([seeds[c]["q"] for c in codes])).sum(axis=0) / wN.sum()
                sa_f, sa_q = share_at_or_above(f), share_at_or_above(q)
                for j, sc in enumerate(og):
                    rows.append(("CAASPP Math", year, "All", "All test-takers", int(published_row(g, year, 1).n_tested), int(sc),
                                 round(float(sa_f[idx[j]]), 5), round(float(sa_q[idx[j]]), 5)))
    out = pd.DataFrame(rows, columns=["test", "year", "group_type", "group", "n", "score", "share_at_or_above", "share_at_or_above_unadjusted"])
    out.to_csv(os.path.join(DATA, "estimated_share_above_cut_schoolmix.csv"), index=False, encoding="utf-8", lineterminator="\n")
    pd.DataFrame(diag).to_csv(os.path.join(DATA, "estimated_diagnostics_schoolmix.csv"), index=False, encoding="utf-8", lineterminator="\n")
    print("wrote", len(out), "rows")


if __name__ == "__main__":
    main(sys.argv[1])
