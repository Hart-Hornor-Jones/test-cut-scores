"""Validation of within-cell normality: the all-student school cells (group
code 1), each a level-anchored normal, pooled and compared with the published
all-student 10-point histogram.  No group constraint or adjustment is involved,
so the differences show what the within-cell assumption gets wrong.

Usage: python check_allstudents_schoolmix.py /path/to/school_levels_grade11_math.csv
Writes data/estimated_diagnostics_schoolmix_allstudents.csv
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

from common import DATA, FLOOR, CEILING, load_published, load_school, histogram, prepare_cells, mixture, share_at_or_above


def main(school_csv):
    s = load_school(school_csv)
    g, h = load_published()
    rows = []
    for year in sorted(CEILING):
        ceiling = CEILING[year]
        grid = np.arange(FLOOR, ceiling + 1)
        cells = prepare_cells(s, year, [1])
        f, _ = mixture(cells, grid, ceiling)
        sa = share_at_or_above(f)
        hy = histogram(h, year)
        tot = hy.n_students.sum()
        hs = np.cumsum(hy.n_students.to_numpy()[::-1])[::-1] / tot
        for i, r in hy.iterrows():
            rows.append(dict(year=year, score=int(r.score_from), histogram_share_at_or_above=round(float(hs[i]), 5),
                             cell_mixture_share_at_or_above=round(float(sa[r.score_from - FLOOR]), 5),
                             difference_pp=round(100 * (sa[r.score_from - FLOOR] - hs[i]), 2),
                             ratio=round(float(sa[r.score_from - FLOOR] / hs[i]), 3) if hs[i] > 0 else np.nan,
                             cells=len(cells)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA, "estimated_diagnostics_schoolmix_allstudents.csv"), index=False, encoding="utf-8", lineterminator="\n")
    piv = out[out.score.isin([2543, 2628, 2718, 2750, 2800, 2860, 2900]) | (out.score == out.groupby("year").score.transform("max"))]
    print(piv.pivot_table(index="year", columns="score", values="ratio").round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
