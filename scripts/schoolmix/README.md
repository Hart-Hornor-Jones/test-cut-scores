# School-mixture estimates (CAASPP grade-11 math)

Scripts behind `data/estimated_*_schoolmix.csv` and METHOD.md §11. They read
the published tables in `data/` and the school-level rows from
<https://github.com/Hart-Hornor-Jones/ca-sbac-math-levels>
(`data/school_levels_grade11_math.csv`), which is passed as a path.

```bash
pip install numpy scipy pandas cvxpy highspy
git clone https://github.com/Hart-Hornor-Jones/ca-sbac-math-levels /path/to/ca-sbac-math-levels
S=/path/to/ca-sbac-math-levels/data/school_levels_grade11_math.csv
cd scripts/schoolmix
python fit_schoolmix.py "$S"                 # point estimates, diagnostics, fit cache (seconds)
python check_allstudents_schoolmix.py "$S"   # within-cell normality check against the histogram
python bounds_schoolmix.py "$S"              # linear-programming bounds (about an hour on four cores)
python compare_schoolmix.py                  # comparison table (CSV and Markdown)
python add_to_page.py                        # adds the estimate to index.html as a menu choice
```

`bounds_schoolmix.py --quick --years=2019,2025` bounds only the top-X% cuts for
the listed years. Every file is opened with `encoding='utf-8'`. The fit cache
(`data/schoolmix_cache/`) is not committed.
