"""Comparison of the school-mixture estimates with the current estimates at the
top 1/5/10/25% cuts: point estimates, bounds, and bound widths.

Usage: python compare_schoolmix.py
Reads data/estimated_share_above_cut*.csv, data/estimated_bounds_schoolmix.csv,
data/estimated_composition_at_standard_cuts.csv and the fit cache.
Writes data/estimated_comparison_schoolmix.csv and prints Markdown tables.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from common import DATA, FLOOR, CEILING, TOP_CUTS, OUTPUT_RACE, OUTPUT_ECON, load_published, published_row

CACHE = os.path.join(DATA, "schoolmix_cache")


def current_point(old, year, gtype, group, score):
    """Share at or above an integer score under the current estimate: exact
    from the composition file where that file has the cut, else log-linear
    interpolation of the 5-point share file."""
    r = old[(old.year == year) & (old.group_type == gtype) & (old.group == group)].sort_values("score")
    sc = r.score.to_numpy(); sh = r.share_at_or_above.to_numpy()
    if score in sc:
        return float(sh[sc == score][0])
    i = np.searchsorted(sc, score) - 1
    s0, s1 = sc[i], sc[i + 1]; a, b = max(sh[i], 1e-9), max(sh[i + 1], 1e-9)
    t = (score - s0) / (s1 - s0)
    return float(np.exp((1 - t) * np.log(a) + t * np.log(b)))


def main():
    g, _ = load_published()
    old = pd.read_csv(os.path.join(DATA, "estimated_share_above_cut.csv"), encoding="utf-8")
    old = old[old.test == "CAASPP Math"]
    comp = pd.read_csv(os.path.join(DATA, "estimated_composition_at_standard_cuts.csv"), encoding="utf-8")
    comp = comp[(comp.test == "CAASPP Math") & (comp.rule == "own_top")]
    bounds = pd.read_csv(os.path.join(DATA, "estimated_bounds_schoolmix.csv"), encoding="utf-8")
    tops = bounds[bounds.cut_label.str.startswith("top")]
    rows = []
    for year in sorted(CEILING):
        grid = np.arange(FLOOR, CEILING[year] + 1)
        fits = {}
        for scheme, outmap, gtype in [("race", OUTPUT_RACE, "Race/ethnicity"), ("econ", OUTPUT_ECON, "Economic status")]:
            z = np.load(os.path.join(CACHE, f"fit_{year}_{scheme}.npz"))
            codes = [int(c) for c in z["codes"]]
            N = {c: published_row(g, year, c).n_tested for c in codes}
            for gname, members in outmap.items():
                ii = [codes.index(c) for c in members]; wN = np.array([N[c] for c in members], float)
                f = (wN[:, None] * z["F"][ii]).sum(axis=0) / wN.sum()
                q = (wN[:, None] * z["Q"][ii]).sum(axis=0) / wN.sum()
                fits[(gtype, gname)] = (np.cumsum(f[::-1])[::-1], np.cumsum(q[::-1])[::-1], wN.sum())
        for x in TOP_CUTS:
            lab = f"top{x:g}"
            tb = tops[(tops.year == year) & (tops.cut_label == lab)]
            if not len(tb):
                continue
            T = int(tb.score.iloc[0])
            # composition denominators (race scheme) under both point estimates
            den_new = sum(fits[("Race/ethnicity", gn)][0][T - FLOOR] * fits[("Race/ethnicity", gn)][2] for gn in OUTPUT_RACE)
            cur_pts = {}
            for gn in OUTPUT_RACE:
                cr = comp[(comp.caaspp_year == year) & (comp.top_percent == x) & (comp.group == gn)]
                cur_pts[gn] = float(cr.share_of_group_selected.iloc[0]) if len(cr) else current_point(old, year, "Race/ethnicity", gn, T)
            den_old = sum(cur_pts[gn] * fits[("Race/ethnicity", gn)][2] for gn in OUTPUT_RACE)
            for gtype, outmap in [("Race/ethnicity", OUTPUT_RACE), ("Economic status", OUTPUT_ECON)]:
                for gname in outmap:
                    sa_f, sa_q, n = fits[(gtype, gname)]
                    new_pt = float(sa_f[T - FLOOR]); seed_pt = float(sa_q[T - FLOOR])
                    cr = comp[(comp.caaspp_year == year) & (comp.top_percent == x) & (comp.group == gname)]
                    if len(cr):
                        cur_pt, src = float(cr.share_of_group_selected.iloc[0]), "composition file"
                    else:
                        cur_pt, src = current_point(old, year, gtype, gname, T), "interpolated from share file"
                    b = tb[(tb.group_type == gtype) & (tb.group == gname)]
                    def bb(method, assumption):
                        r = b[(b.method == method) & (b.assumption == assumption)]
                        return (float(r.share_lo.iloc[0]), float(r.share_hi.iloc[0])) if len(r) else (np.nan, np.nan)
                    cb = bb("current", "bell"); cn = bb("current", "none"); sb = bb("schoolmix", "bell"); sn = bb("schoolmix", "none")
                    d = dict(year=year, cut=lab, top_percent=x, cut_score=T, group_type=gtype, group=gname, n=int(n),
                             current_point=round(cur_pt, 5), current_point_source=src,
                             schoolmix_point=round(new_pt, 5), schoolmix_point_unadjusted=round(seed_pt, 5),
                             point_change_pp=round(100 * (new_pt - cur_pt), 2),
                             point_change_relative=round((new_pt - cur_pt) / cur_pt, 3) if cur_pt > 0 else np.nan,
                             current_bell_lo=round(cb[0], 5), current_bell_hi=round(cb[1], 5), current_bell_width_pp=round(100 * (cb[1] - cb[0]), 2),
                             current_none_lo=round(cn[0], 5), current_none_hi=round(cn[1], 5), current_none_width_pp=round(100 * (cn[1] - cn[0]), 2),
                             schoolmix_lo=round(sb[0], 5), schoolmix_hi=round(sb[1], 5), schoolmix_width_pp=round(100 * (sb[1] - sb[0]), 2),
                             schoolmix_none_lo=round(sn[0], 5), schoolmix_none_hi=round(sn[1], 5), schoolmix_none_width_pp=round(100 * (sn[1] - sn[0]), 2),
                             width_ratio_schoolmix_vs_current_bell=round((sb[1] - sb[0]) / (cb[1] - cb[0]), 3) if cb[1] > cb[0] else np.nan,
                             share_without_shape_constraint=float(b[(b.method == "schoolmix") & (b.assumption == "bell")].share_without_shape_constraint.iloc[0]) if len(b) else np.nan)
                    if gtype == "Race/ethnicity":
                        d["current_share_of_selected"] = round(cur_pt * n / den_old, 4)
                        d["schoolmix_share_of_selected"] = round(new_pt * n / den_new, 4)
                    rows.append(d)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA, "estimated_comparison_schoolmix.csv"), index=False, encoding="utf-8", lineterminator="\n")
    print(markdown(out))


def pct(v, d=1):
    return f"{100 * v:.{d}f}%"


def markdown(out):
    big = ["Asian", "Black", "Hispanic/Latino", "White", "Two or more"]
    lines = []
    # 1. bound widths: median ratio by cut, race groups (the five largest) and economic groups
    lines.append("### How much the bounds narrow (bell-shape bounds, width of the school-mixture bounds as a share of the current width; median over years)\n")
    lines.append("| Group | " + " | ".join(f"top {x:g}%" for x in TOP_CUTS) + " |")
    lines.append("|---|" + "---|" * len(TOP_CUTS))
    for gname in big + ["American Indian", "Pacific Islander", "Economically disadvantaged", "Not economically disadvantaged"]:
        cells = []
        for x in TOP_CUTS:
            r = out[(out.group == gname) & (out.top_percent == x)]
            cells.append(f"{r.width_ratio_schoolmix_vs_current_bell.median():.2f} ({r.current_bell_width_pp.median():.1f} → {r.schoolmix_width_pp.median():.1f} pp)")
        lines.append(f"| {gname} | " + " | ".join(cells) + " |")
    lines.append("")
    # 2. point moves: by cut, median and max absolute change for the five largest race groups
    lines.append("### Whether the point estimates move (share of the group at or above the cut, school mixture minus current, percentage points; all years)\n")
    lines.append("| Group | " + " | ".join(f"top {x:g}%: median / largest" for x in TOP_CUTS) + " |")
    lines.append("|---|" + "---|" * len(TOP_CUTS))
    for gname in big + ["Economically disadvantaged", "Not economically disadvantaged"]:
        cells = []
        for x in TOP_CUTS:
            r = out[(out.group == gname) & (out.top_percent == x)].point_change_pp
            cells.append(f"{r.median():+.2f} / {r.loc[r.abs().idxmax()]:+.2f}")
        lines.append(f"| {gname} | " + " | ".join(cells) + " |")
    lines.append("")
    # 3. detail for 2019 and 2025
    for year in (2019, 2025):
        lines.append(f"### {year}: point estimate and bounds, share of the group at or above the cut (current → school mixture)\n")
        lines.append("| Cut (score) | Group | Point | Bell-shape bounds | No-assumption bounds | Share of selected (race groups) |")
        lines.append("|---|---|---|---|---|---|")
        for x in TOP_CUTS:
            for gname in big + ["Economically disadvantaged", "Not economically disadvantaged"]:
                r = out[(out.year == year) & (out.group == gname) & (out.top_percent == x)]
                if not len(r):
                    continue
                r = r.iloc[0]
                comp = f"{pct(r.current_share_of_selected)} → {pct(r.schoolmix_share_of_selected)}" if not pd.isna(r.get("current_share_of_selected", np.nan)) else ""
                lines.append(f"| top {x:g}% ({r.cut_score}) | {gname} | {pct(r.current_point)} → {pct(r.schoolmix_point)} | "
                             f"{pct(r.current_bell_lo)}–{pct(r.current_bell_hi)} → {pct(r.schoolmix_lo)}–{pct(r.schoolmix_hi)} | "
                             f"{pct(r.current_none_lo)}–{pct(r.current_none_hi)} → {pct(r.schoolmix_none_lo)}–{pct(r.schoolmix_none_hi)} | {comp} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
