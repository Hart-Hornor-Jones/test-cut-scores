# Cut scores and who they select

An interactive page comparing what a purely test-based admissions cut would select in California, by race/ethnicity or gender, under the SAT and under CAASPP (the state's grade-11 Smarter Balanced test).

Open `index.html`. Pick a test, a cohort, and a grouping (race/ethnicity, gender, or — for CAASPP — economic status), hover along the left chart to try a cut score, click to hold it. The left chart shows the share of each group scoring at or above the cut; the right chart shows who is above it.

## Data

`data/published_*` are the published numbers, copied from the source reports.

| File | What it holds |
|---|---|
| `published_caaspp_grade11_by_group.csv` | Grade 11, 2014-15 to 2024-25: N, mean, SD and the four achievement-level shares by race/ethnicity, gender, economic status, and race × economic status. CDE technical reports, Appendix 7. |
| `published_caaspp_grade11_score_histogram.csv` | Grade 11, all students: counts in 10-point score bins. Same source. |
| `published_sat_california_bands_by_group.csv` | Classes of 2018–2026: share of each race/ethnicity and gender group in six 200-point Total-score bands. College Board SAT Suite annual reports, California. |
| `published_sat_california_means_by_group.csv` | Classes of 2017–2026: N and mean scores by group. Same source. |
| `published_sat_california_bands_all_takers.csv` | Classes of 2017–2025: exact counts in the six bands, all takers. Same source. |
| `published_sat_california_summary_all_takers.csv` | Classes of 2017–2026: mean, SD and quartiles, all takers. Same source. |

`data/estimated_*` are reconstructions, not published figures.

| File | What it holds |
|---|---|
| `estimated_share_above_cut.csv` | For each test, year, grouping and group: the estimated share scoring at or above each score. This is what the page draws. |
| `estimated_composition_at_standard_cuts.csv` | The composition of the selected group at standard cuts (top 1% to 50% of takers, fixed seat counts, and the published band edges), with each group's selection rate and its representation relative to all public-school juniors. |

How the estimates were made, and how uncertain they are, is in `METHOD.md`. In short: each group's distribution is the smoothest bell-shaped curve that reproduces every published number for that group and adds up, across groups, to the published all-student distribution.

## School-mixture estimates for CAASPP math (alternative)

`data/estimated_*_schoolmix.csv` are a second set of estimates for CAASPP grade-11 mathematics only, built from school-level results (one near-normal component per school × student group, from the public research-file rows in [ca-sbac-math-levels](https://github.com/Hart-Hornor-Jones/ca-sbac-math-levels)) and reconciled to the same published statewide figures and histogram as the default estimates. They are not the default; the page shows them as a separate choice in the test menu ("CAASPP grade 11 Math — school-mixture estimate"), without a gender grouping.

| File | What it holds |
|---|---|
| `estimated_share_above_cut_schoolmix.csv` | Same layout as `estimated_share_above_cut.csv`, with the unreconciled mixture as an extra column. |
| `estimated_bounds_schoolmix.csv` | Min/max bounds on the share at or above the level cuts, the top-1/5/10/25% cuts and the ceiling, for the school mixture and for the current method on the same grid. |
| `estimated_comparison_schoolmix.csv` | Point estimates and bounds under both methods at the top-1/5/10/25% cuts, every year and group. |
| `estimated_diagnostics_schoolmix*.csv` | Coverage of each group by unsuppressed cells, the size of the reconciliation, and the all-student check of within-cell normality. |

How the mixture is built, what it assumes and where it fails is in `METHOD.md`, §11; the scripts are in `scripts/schoolmix/`. In short, compared with the default estimates:

- **Bounds.** At the top-5%, top-10% and top-25% cuts the bell-shape bounds on a group's share above the cut narrow to between a tenth and a half of their current width, typically a fifth to a third, for Asian, Hispanic/Latino, White, Two-or-more and both economic-status groups (2015–2025; for example the 2025 top-10% bounds for Asian students go from 13.1%–40.0% to 26.6%–33.2%, and for Hispanic/Latino students from 1.5%–5.7% to 3.4%–4.4%). At the top 1% they narrow to about 40% of their width. Black students' bounds narrow less (about 80% of the group is in unsuppressed cells), and American Indian and Pacific Islander students' bounds hardly at all (about 14% is).
- **Point estimates.** They move little: never by more than 0.5 percentage points below the top 5% for Hispanic/Latino, Black, White or economically disadvantaged students, and by at most 1.2 points for any group. At the top 1% and 5% the mixture gives a slightly less Asian and slightly more Hispanic/Latino, White and Black selection than the statewide bell curves (2019 top 5%: Asian share of the selected 48.7% → 45.5%, Hispanic/Latino 12.8% → 15.5%).
- **Reconciliation.** The mixture reconciles with every published statewide figure within rounding and with the histogram exactly, in every year and grouping, without relaxing any constraint. Its main known weakness is that a normal fitted at the level cuts overstates the very top of the scale before reconciliation (by a fifth to a half of the share at or above 2860), so the top-1% rows are the least settled.

The full comparison, every year and group, is in `data/estimated_comparison_schoolmix.csv`; the summary tables are in `METHOD.md`, §11.4.

## Notes

- CAASPP covers every tested public-school junior. The SAT covers students in the graduating class who took it: 63% of juniors in the class of 2020, about 30% from 2023 on.
- CAASPP spring of year Y is matched to the SAT class of Y+1.
- Filipino students are counted as Asian, as the College Board does. SAT takers with no reported race (or gender) are left out of that grouping. The SAT has no economic-status distributions.
- CAASPP 2017-18 is not included (the report was not in hand). CAASPP 2014-15 and 2015-16 have no SAT counterpart, because the College Board did not publish band shares by group before the class of 2018.
