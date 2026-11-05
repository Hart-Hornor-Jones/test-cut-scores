"""Linear-programming bounds on the share of each group at or above a score,
for the school-mixture model and (recomputed on the same grid) for the current
statewide model.

Usage: python bounds_schoolmix.py /path/to/school_levels_grade11_math.csv [--quick] [--years 2019,2025]

Writes data/estimated_bounds_schoolmix.csv.
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import highspy
from multiprocessing import Pool

from common import (DATA, FLOOR, CUTS, CEILING, TOP_CUTS, RACE_CODES, ECON_CODES, OUTPUT_RACE, OUTPUT_ECON,
                    load_published, load_school, published_row, histogram, prepare_cells, lp_bins)

CELL_MEAN_TOL = 15.5     # score points of slack on a cluster mean under 'none': the bins
                         # below Level 2 are 30 points wide, so a mean near the floor
                         # cannot be hit more closely
GROUP_MEAN_EXTRA = 0.5   # same, for the group mean
GROUP_SD_TOL = 2.0       # METHOD.md: fitted SDs match within about 1.5; +0.5 for the bins
ROUND = 0.005            # published whole percents, as a fraction
CLUSTER_MU, CLUSTER_SD = 10, 20
SOLVER = "ipm"


def top_cut_scores(year):
    """Cut scores for the top-X% rules: from the existing composition file where
    it has them, otherwise from the school-mixture all-student distribution
    (which reproduces the histogram) with the same rule."""
    comp = pd.read_csv(os.path.join(DATA, "estimated_composition_at_standard_cuts.csv"), encoding="utf-8")
    comp = comp[(comp.test == "CAASPP Math") & (comp.rule == "own_top") & (comp.caaspp_year == year)]
    out = {}
    for x in TOP_CUTS:
        r = comp[comp.top_percent == x]
        if len(r):
            out[x] = (int(r.cut_score.iloc[0]), "composition file")
    if len(out) < len(TOP_CUTS):
        z = np.load(os.path.join(DATA, "schoolmix_cache", f"fit_{year}_race.npz"))
        g, _ = load_published()
        codes = list(z["codes"]); grid = z["grid"]
        wN = np.array([published_row(g, year, int(c)).n_tested for c in codes], float)
        f = (wN[:, None] * z["F"]).sum(axis=0) / wN.sum()
        sa = np.cumsum(f[::-1])[::-1]
        for x in TOP_CUTS:
            if x not in out:
                t = grid[np.argmax(sa <= x / 100)]       # smallest score with share at or above <= X%
                out[x] = (int(t), "rule applied to school-mixture all-student distribution")
    return out


SMALL_CELL = 50          # cells with fewer students are pooled more coarsely


def clusters_for(cells: pd.DataFrame):
    """Pool cells with the same group code, similar mean and similar fitted SD.
    Cells with fewer than SMALL_CELL students are pooled on a four-times
    coarser grid so that their lumpy level shares average out."""
    c = cells.copy()
    small = c.students_with_scores < SMALL_CELL
    c["kmu"] = np.where(small, (c.mean_scale_score / (4 * CLUSTER_MU)).round() * 4, (c.mean_scale_score / CLUSTER_MU).round()).astype(int)
    c["ksd"] = np.where(small, (c.sd_fit / (4 * CLUSTER_SD)).round() * 4, (c.sd_fit / CLUSTER_SD).round()).astype(int)
    c["small"] = small
    agg = c.groupby(["small", "kmu", "ksd"]).apply(lambda d: pd.Series(dict(
        n=d.students_with_scores.sum(), cells=len(d),
        mu=np.average(d.mean_scale_score, weights=d.students_with_scores),
        sd=np.average(d.sd_fit, weights=d.students_with_scores),
        p1=np.average(d.p1, weights=d.students_with_scores), p2=np.average(d.p2, weights=d.students_with_scores),
        p3=np.average(d.p3, weights=d.students_with_scores), p4=np.average(d.p4, weights=d.students_with_scores))),
        include_groups=False).reset_index(drop=True)
    return agg


def shape_feasible(comp, bins, cell_round, assumption="bell"):
    """Can this component's level shares be met together with its extra
    constraint (the near-normal envelope under 'bell', the published mean
    under 'none')?  When not, the component keeps only its level shares."""
    B = len(bins)
    L = LPBuilder()
    add_component(L, lambda k, b: b, 0, comp, bins, assumption, cell_round)
    A = sp.coo_matrix((L.vals, (L.rows, L.cols)), shape=(L.nrows, B)).tocsc()
    lp = highspy.HighsLp()
    lp.num_col_ = B; lp.num_row_ = L.nrows
    lp.col_cost_ = np.zeros(B); lp.col_lower_ = np.zeros(B); lp.col_upper_ = np.full(B, np.inf)
    lp.row_lower_ = np.array(L.rlo, float); lp.row_upper_ = np.array(L.rhi, float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32); lp.a_matrix_.index_ = A.indices.astype(np.int32); lp.a_matrix_.value_ = A.data
    hs = highspy.Highs(); hs.setOptionValue("output_flag", False); hs.setOptionValue("threads", 1)
    hs.passModel(lp); hs.run()
    return hs.getModelStatus() == highspy.HighsModelStatus.kOptimal


class LPBuilder:
    def __init__(self):
        self.rows, self.cols, self.vals = [], [], []
        self.rlo, self.rhi = [], []
        self.nrows = 0

    def add(self, coef: dict, lo, hi):
        r = self.nrows
        for c, v in coef.items():
            self.rows.append(r); self.cols.append(c); self.vals.append(v)
        self.rlo.append(lo); self.rhi.append(hi)
        self.nrows += 1


ENVELOPE_MIN_REL = 0.10   # the SD range is never narrower than +-10% of the fitted SD


def sd_range(mu, sd_fit, p):
    """SDs implied one cut at a time by the component's shares at or above
    each level cut (mean fixed).  The range of these, widened to at least
    +-ENVELOPE_MIN_REL of the fitted SD, is the set of normal shapes the LP
    allows inside each level."""
    from scipy.stats import norm as _n
    c = [p[1] + p[2] + p[3], p[2] + p[3], p[3]]
    implied = []
    for cut, cj in zip(CUTS, c):
        if 0.002 < cj < 0.998 and abs(cut - mu) > 5:
            z = _n.isf(cj)
            if z * (cut - mu) > 0:
                implied.append((cut - mu) / z)
    lo = min([sd_fit * (1 - ENVELOPE_MIN_REL)] + implied)
    hi = max([sd_fit * (1 + ENVELOPE_MIN_REL)] + implied)
    return max(lo, 10.0), min(hi, 400.0)


def envelope_rows(L, var, k, mu, sd_lo, sd_hi, p, bins):
    """Within each level, the share at or above each bin start (as a fraction
    of the level's mass) must lie between the conditional normal tails over
    sd in [sd_lo, sd_hi]."""
    from scipy.stats import norm as _n
    B = len(bins)
    lo = bins.lo.to_numpy(); hi = bins.hi.to_numpy()
    lvl = np.zeros(B, int)
    for c in CUTS:
        lvl += (lo >= c)
    level_lo = [FLOOR] + list(CUTS)
    level_hi = list(CUTS) + [int(hi.max()) + 1]
    sds = np.linspace(sd_lo, sd_hi, 9)
    for lev in range(4):
        bs = [b for b in range(B) if lvl[b] == lev]
        if p[lev] <= 0 or len(bs) < 2:
            continue
        a, z = level_lo[lev], level_hi[lev]
        # mass in the level under each sd (floor and ceiling piles included)
        cdf_a = _n.cdf((a - 0.5 - mu) / sds) if lev > 0 else np.zeros(len(sds))
        cdf_z = _n.cdf((z - 0.5 - mu) / sds) if lev < 3 else np.ones(len(sds))
        tot = cdf_z - cdf_a
        for b in bs[1:]:
            cdf_b = _n.cdf((lo[b] - 0.5 - mu) / sds)
            cond = (cdf_z - cdf_b) / np.maximum(tot, 1e-300)
            cond = np.clip(cond, 0, 1)
            coef = {var(k, bb): 1.0 for bb in bs if lo[bb] >= lo[b]}
            L.add(coef, p[lev] * cond.min(), p[lev] * cond.max())


def add_component(L, var, k, comp, bins, assumption, cell_round):
    """Constraints on one component (a cluster of cells, or a remainder)."""
    code, m, kind, mu, sd, p = comp[:6]
    shape = comp[6] if len(comp) > 6 else True
    B = len(bins)
    lo = bins.lo.to_numpy(); mid = bins.mid.to_numpy(); width = bins.width.to_numpy()
    ceiling = int(lo.max())
    is_ceiling = (lo == ceiling)
    lvl = np.zeros(B, int)
    for c in CUTS:
        lvl += (lo >= c)
    # variables are the component's distribution in units of its own mass
    L.add({var(k, b): 1.0 for b in range(B)}, 1.0, 1.0)
    if kind == "cell":
        for lev in range(4):
            L.add({var(k, b): 1.0 for b in range(B) if lvl[b] == lev}, max(p[lev] - cell_round, 0), p[lev] + cell_round)
        if assumption == "none" and shape:
            # location inside the levels comes from the published cell mean when
            # no shape is assumed; under the envelope it comes from the normal
            L.add({var(k, b): mid[b] for b in range(B)}, mu - 0.05 - CELL_MEAN_TOL, mu + 0.05 + CELL_MEAN_TOL)
    if assumption == "bell" and shape and kind == "cell":
        s_lo, s_hi = sd_range(mu, sd, p)
        envelope_rows(L, var, k, mu, s_lo, s_hi, p, bins)
    if assumption == "bell" and shape and kind != "cell":
        reg = [b for b in range(B) if not is_ceiling[b]]
        for i in range(len(reg) - 1):
            b0, b1 = reg[i], reg[i + 1]
            if lo[b0] >= mu + sd:
                L.add({var(k, b1): 1.0 / width[b1], var(k, b0): -1.0 / width[b0]}, -np.inf, 0.0)
        for i in range(1, len(reg) - 1):
            bm, b0, bp = reg[i - 1], reg[i], reg[i + 1]
            if lo[bm] >= mu + 1.5 * sd:
                dp = mid[bp] - mid[b0]; dm = mid[b0] - mid[bm]
                coef = {var(k, bp): 1.0 / width[bp] / dp, var(k, b0): -1.0 / width[b0] / dp - 1.0 / width[b0] / dm,
                        var(k, bm): 1.0 / width[bm] / dm}
                L.add(coef, 0.0, np.inf)
        bc = int(np.where(is_ceiling)[0][0]); bl = reg[-1]
        if lo[bl] >= mu + sd:
            L.add({var(k, bc): 1.0, var(k, bl): -sd / width[bl]}, -np.inf, 0.0)


def build_lp(year, scheme, method, assumption, s, g, h, extra_edges, cell_round):
    """Returns (highs, info) with the LP built and the minimal-relaxation tau solved."""
    codes = list(RACE_CODES) if scheme == "race" else list(ECON_CODES)
    bins = lp_bins(year, extra_edges, coarse_low=True)
    ceiling = CEILING[year]
    B = len(bins)
    lo = bins.lo.to_numpy(); hi = bins.hi.to_numpy(); mid = bins.mid.to_numpy(); width = bins.width.to_numpy()
    is_ceiling = (lo == ceiling)
    lvl = np.zeros(B, int)
    for c in CUTS:
        lvl += (lo >= c)
    hy = histogram(h, year)
    hshare = hy.n_students.to_numpy(float) / hy.n_students.sum()
    hist_of_bin = np.array([int(np.where((hy.score_from <= l) & (hy.score_to >= l))[0][0]) for l in lo])
    # histogram constraints: single 10-point bins from Level 2 up; below Level 2
    # the bins are pooled in threes (a relaxation that keeps the LP small)
    hgroup = np.array([j if hy.score_from[j] >= CUTS[0] else -(1 + j // 3) for j in range(len(hy))])
    pub_all = published_row(g, year, 1)
    n_all_cells = s[(s.year == year) & s.known & (s.group_code == 1)].students_with_scores.sum()

    comps = []   # (code, share m, kind, mu, sd, p(4) or None)
    for code in codes:
        pub = published_row(g, year, code)
        if method == "schoolmix":
            cells = prepare_cells(s, year, [code])
            n_known = cells.students_with_scores.sum()
            n_target = pub.n_tested / pub_all.n_tested * n_all_cells
            if n_known > n_target:
                n_target = n_known
            n_rem = n_target - n_known
            if len(cells):
                cl = clusters_for(cells)
                for _, r in cl.iterrows():
                    comp = (code, r.n / n_target, "cell", r.mu, r.sd, np.array([r.p1, r.p2, r.p3, r.p4]), True)
                    if not shape_feasible(comp, bins, cell_round, assumption):
                        comp = comp[:6] + (False,)
                    comps.append(comp)
            if n_rem > 0:
                comps.append((code, n_rem / n_target, "remainder", pub.mean_scale_score, pub.sd_scale_score, None, True))
        else:
            comps.append((code, 1.0, "remainder", pub.mean_scale_score, pub.sd_scale_score, None, True))
    K = len(comps)
    nv = K * B + 1
    TAU = nv - 1
    col_lo = np.zeros(nv); col_hi = np.ones(nv); col_hi[TAU] = np.inf
    L = LPBuilder()

    def var(k, b):
        return k * B + b

    for k, comp in enumerate(comps):
        add_component(L, var, k, comp, bins, assumption, cell_round)

    # group-level published constraints, with tau relaxation (tau in fractions)
    wN = {code: published_row(g, year, code).n_tested for code in codes}
    W = sum(wN.values())
    for code in codes:
        pub = published_row(g, year, code)
        ks = [k for k, c in enumerate(comps) if c[0] == code]
        mass = {k: comps[k][1] for k in ks}
        p = np.array([pub.pct_level1_not_met, pub.pct_level2_nearly_met, pub.pct_level3_met, pub.pct_level4_exceeded]) / 100
        for lev in range(4):
            coef = {var(k, b): mass[k] for k in ks for b in range(B) if lvl[b] == lev}
            c1 = dict(coef); c1[TAU] = -1.0; L.add(c1, -np.inf, p[lev] + ROUND)
            c2 = dict(coef); c2[TAU] = 1.0; L.add(c2, p[lev] - ROUND, np.inf)
        coef = {var(k, b): mass[k] * mid[b] for k in ks for b in range(B)}
        c1 = dict(coef); c1[TAU] = -100.0; L.add(c1, -np.inf, pub.mean_scale_score + 0.5 + GROUP_MEAN_EXTRA)
        c2 = dict(coef); c2[TAU] = 100.0; L.add(c2, pub.mean_scale_score - 0.5 - GROUP_MEAN_EXTRA, np.inf)
        m2 = mid ** 2 + width ** 2 / 12.0
        m2_lo = (pub.sd_scale_score - GROUP_SD_TOL) ** 2 + (pub.mean_scale_score - 0.5) ** 2
        m2_hi = (pub.sd_scale_score + GROUP_SD_TOL) ** 2 + (pub.mean_scale_score + 0.5) ** 2
        coef = {var(k, b): mass[k] * m2[b] for k in ks for b in range(B)}
        c1 = dict(coef); c1[TAU] = -200.0 * pub.sd_scale_score; L.add(c1, -np.inf, m2_hi)
        c2 = dict(coef); c2[TAU] = 200.0 * pub.sd_scale_score; L.add(c2, m2_lo, np.inf)
    # histogram
    for gj in sorted(set(hgroup)):
        js = [j for j in range(len(hy)) if hgroup[j] == gj]
        target = float(sum(hshare[j] for j in js))
        coef = {}
        for k, c in enumerate(comps):
            wgt = wN[c[0]] / W * c[1]
            for b in range(B):
                if hist_of_bin[b] in js:
                    coef[var(k, b)] = wgt
        c1 = dict(coef); c1[TAU] = -1.0; L.add(c1, -np.inf, target)
        c2 = dict(coef); c2[TAU] = 1.0; L.add(c2, target, np.inf)

    A = sp.coo_matrix((L.vals, (L.rows, L.cols)), shape=(L.nrows, nv)).tocsc()
    lp = highspy.HighsLp()
    lp.num_col_ = nv; lp.num_row_ = L.nrows
    lp.col_cost_ = np.zeros(nv); lp.col_cost_[TAU] = 1.0
    lp.col_lower_ = col_lo; lp.col_upper_ = col_hi
    lp.row_lower_ = np.array(L.rlo, float); lp.row_upper_ = np.array(L.rhi, float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32); lp.a_matrix_.index_ = A.indices.astype(np.int32); lp.a_matrix_.value_ = A.data
    hs = highspy.Highs()
    hs.setOptionValue("output_flag", False)
    hs.setOptionValue("threads", 1)
    hs.setOptionValue("solver", SOLVER)
    if SOLVER == "ipm":
        hs.setOptionValue("run_crossover", "off")
    hs.passModel(lp)
    hs.run()
    st = hs.getModelStatus()
    if st != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"{year} {scheme} {method} {assumption}: tau LP status {hs.modelStatusToString(st)}")
    tau = hs.getSolution().col_value[TAU]
    tau_fix = tau + 1e-6 if tau > 1e-7 else 0.0
    hs.changeColsBounds(1, np.array([TAU], dtype=np.int32), np.array([tau_fix]), np.array([tau_fix]))
    hs.changeColsCost(1, np.array([TAU], dtype=np.int32), np.array([0.0]))
    noshape = {}
    for code in codes:
        noshape[code] = sum(c[1] for c in comps if c[0] == code and c[2] == "cell" and not c[6])
    info = dict(comps=comps, bins=bins, B=B, nv=nv, TAU=TAU, tau=tau, codes=codes, wN=wN, var=var, nrows=L.nrows, noshape=noshape)
    return hs, info


def solve_bounds(hs, info, outmap, cut_scores):
    """For each output group and cut score, min and max of the share at or above."""
    B = info["B"]; comps = info["comps"]; var = info["var"]; lo = info["bins"].lo.to_numpy(); wN = info["wN"]
    res = []
    for gname, members in outmap.items():
        NG = sum(wN[c] for c in members)
        for T in cut_scores:
            idx, cost = [], []
            for k, c in enumerate(comps):
                if c[0] in members:
                    wgt = wN[c[0]] / NG * c[1]
                    for b in range(B):
                        if lo[b] >= T:
                            idx.append(var(k, b)); cost.append(wgt)
            idx = np.array(idx, dtype=np.int32); cost = np.array(cost)
            allcols = np.arange(info["nv"], dtype=np.int32)
            hs.changeColsCost(info["nv"], allcols, np.zeros(info["nv"]))
            hs.changeColsCost(len(idx), idx, cost)
            out = {}
            for sign, name in [(1.0, "lo"), (-1.0, "hi")]:
                hs.changeColsCost(len(idx), idx, sign * cost)
                hs.run()
                st = hs.getModelStatus()
                if st != highspy.HighsModelStatus.kOptimal:
                    out[name] = np.nan
                    print("  warning:", gname, T, name, hs.modelStatusToString(st), flush=True)
                else:
                    out[name] = sign * hs.getInfo().objective_function_value
            res.append((gname, int(T), out["lo"], out["hi"]))
    return res


def run_one(args):
    year, scheme, method, assumption, school_csv, quick = args
    t0 = time.time()
    cached = os.path.join(DATA, "schoolmix_cache", f"bounds_{year}_{scheme}_{method}_{assumption}.csv")
    if os.path.exists(cached) and not quick:
        print(f"cached {year} {scheme} {method} {assumption}", flush=True)
        return pd.read_csv(cached, encoding="utf-8").to_dict("records")
    s = load_school(school_csv)
    g, h = load_published()
    tops = top_cut_scores(year)
    cell_round = 0.005 if year in (2015, 2016) else 0.0005
    ceiling = CEILING[year]
    cut_scores = sorted(set(CUTS) | {t for t, _ in tops.values()} | {ceiling})
    if quick:
        cut_scores = sorted({t for t, _ in tops.values()})
    hs, info = build_lp(year, scheme, method, assumption, s, g, h, [t for t, _ in tops.values()], cell_round)
    outmap = OUTPUT_RACE if scheme == "race" else OUTPUT_ECON
    res = solve_bounds(hs, info, outmap, cut_scores)
    gtype = "Race/ethnicity" if scheme == "race" else "Economic status"
    label = {CUTS[0]: "level2", CUTS[1]: "level3", CUTS[2]: "level4", ceiling: "ceiling"}
    label.update({t: f"top{x:g}" for x, (t, _) in tops.items()})   # a top-X% cut at the ceiling is labeled topX
    rows = [dict(test="CAASPP Math", year=year, group_type=gtype, group=gname, score=T,
                 cut_label=label.get(T, "grid"), method=method, assumption=assumption,
                 share_lo=round(lo_, 5), share_hi=round(hi_, 5), tau_pct=round(100 * info["tau"], 4),
                 lp_components=len(info["comps"]), lp_rows=info["nrows"],
                 share_without_shape_constraint=round(sum(info["noshape"][c] * info["wN"][c] for c in outmap[gname]) / sum(info["wN"][c] for c in outmap[gname]), 4))
            for gname, T, lo_, hi_ in res]
    print(f"done {year} {scheme} {method} {assumption}: {len(info['comps'])} components, {info['nrows']} rows, tau={100*info['tau']:.4f} pts, {len(res)} objectives x2, {time.time()-t0:.0f}s", flush=True)
    cache = os.path.join(DATA, "schoolmix_cache")
    os.makedirs(cache, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(cache, f"bounds_{year}_{scheme}_{method}_{assumption}.csv"), index=False, encoding="utf-8", lineterminator="\n")
    return rows


def main():
    school_csv = sys.argv[1]
    quick = "--quick" in sys.argv
    years = sorted(CEILING)
    for a in sys.argv:
        if a.startswith("--years"):
            years = [int(y) for y in a.split("=")[1].split(",")]
    jobs = []
    for year in years:
        for scheme in ("race", "econ"):
            for method, assumption in [("schoolmix", "bell"), ("current", "bell"), ("current", "none"), ("schoolmix", "none")]:
                jobs.append((year, scheme, method, assumption, school_csv, quick))
    # heavy jobs first
    jobs.sort(key=lambda j: (j[2] != "schoolmix", j[1] != "race"))
    with Pool(4) as pool:
        results = pool.map(run_one, jobs, chunksize=1)
    rows = [r for rr in results for r in rr]
    out = pd.DataFrame(rows).sort_values(["year", "group_type", "group", "score", "method", "assumption"])
    path = os.path.join(DATA, "estimated_bounds_schoolmix.csv")
    out.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    print("wrote", len(out), "rows to", path)


if __name__ == "__main__":
    main()
