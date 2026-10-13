# Cut scores and who they select

An interactive page comparing what a purely test-based admissions cut would select in California, by race/ethnicity, under the SAT and under CAASPP (the state's grade-11 Smarter Balanced test).

Open `index.html`. Pick a test and a cohort, hover along the left chart to try a cut score, click to hold it. The left chart shows the share of each group scoring at or above the cut; the right chart shows who is above it.

## Data

`data/published_*` are the published numbers, copied from the source reports.

| File | What it holds |
|---|---|
| `published_caaspp_grade11_by_group.csv` | Grade 11, 2014-15 to 2024-25: N, mean, SD and the four achievement-level shares by race/ethnicity, economic status, and race × economic status. CDE technical reports, Appendix 7. |
| `published_caaspp_grade11_score_histogram.csv` | Grade 11, all students: counts in 10-point score bins. Same source. |
| `published_sat_california_bands_by_group.csv` | Classes of 2018–2026: share of each race/ethnicity group in six 200-point Total-score bands. College Board SAT Suite annual reports, California. |
| `published_sat_california_means_by_group.csv` | Classes of 2017–2026: N and mean scores by group. Same source. |
| `published_sat_california_bands_all_takers.csv` | Classes of 2017–2025: exact counts in the six bands, all takers. Same source. |
| `published_sat_california_summary_all_takers.csv` | Classes of 2017–2026: mean, SD and quartiles, all takers. Same source. |

`data/estimated_*` are reconstructions, not published figures.

| File | What it holds |
|---|---|
| `estimated_share_above_cut.csv` | For each test, year and group: the estimated share scoring at or above each score. This is what the page draws. |
| `estimated_composition_at_standard_cuts.csv` | The composition of the selected group at standard cuts (top 1% to 50% of takers, fixed seat counts, and the published band edges), with each group's selection rate and its representation relative to all public-school juniors. |

How the estimates were made, and how uncertain they are, is in `METHOD.md`. In short: each group's distribution is the smoothest bell-shaped curve that reproduces every published number for that group and adds up, across groups, to the published all-student distribution.

## Notes

- CAASPP covers every tested public-school junior. The SAT covers students in the graduating class who took it: 63% of juniors in the class of 2020, about 30% from 2023 on.
- CAASPP spring of year Y is matched to the SAT class of Y+1.
- Filipino students are counted as Asian, as the College Board does. SAT takers with no reported race are left out.
- CAASPP 2017-18 is not included (the report was not in hand). CAASPP 2014-15 and 2015-16 have no SAT counterpart, because the College Board did not publish band shares by group before the class of 2018.
