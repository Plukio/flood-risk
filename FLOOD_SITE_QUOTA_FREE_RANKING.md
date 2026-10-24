# Bangkok first-in-year site ranking without a discovery reserve

**24 October 2026.** The earlier paper used a fixed 20% discovery reserve within a hypothetical top-10% inspection shortlist. That reserve was an illustrative policy choice, not a BMA staffing measurement or a source-selected optimum. This check removes that allocation from **model selection and the reported comparison**.

**Full-queue follow-up:** [The all-cell capture curves](FLOOD_SITE_ALL_CELL_CURVES.md) show no difference at the top 10% of all district cells: 38.32% capture of all reported sites and 3.50% capture of emerging sites for both rankers. Better ordering *within* unfamiliar cells has not yet delivered an improvement in a tight mixed queue.

## Evaluation

- Each day and district, the two rankers order the *same* candidate cells with no report in that cell during the preceding 365 days. No fraction of the full district shortlist is set aside for them.
- The outcome is a first-in-year raw Traffy report-site medoid with no earlier report within 250 m. Of 471 Q4 2025 such site events, **393** fall in a cell with zero own-cell report history and can enter this candidate ranking. The other 78 need a different all-cell comparison and are excluded from this conditional analysis.
- The source districts' January–September 2025 events choose the CNN training arm and history/CNN blend by **mean percentile rank of new sites**. This criterion has no cutoff or reserve. Q4 2025 districts excluded from fitting and selection are then evaluated.
- A percentile of 0.65 means a reported site scores above about 65% of eligible unfamiliar cells in its district that day. It is a ranking measure, not a probability that the site floods.

The quota-free source selection chose the **new-site-only CNN arm in all five folds**. Q4 first-in-year sites had mean rank percentile **0.6570** under that CNN/history blend versus **0.6017** under nearby history: **+5.53 percentile points**. A paired district bootstrap interval for the difference is **+2.75 to +8.00 points**. This interval does not include the uncertainty from repeated Q4 inspection or target design.

| Share of unfamiliar candidate cells reviewed | Nearby history: first-in-year sites found | CNN/history blend: sites found |
| ---: | ---: | ---: |
| 1% | 1.69% | 2.57% |
| 2% | 3.32% | 5.03% |
| 5% | 6.82% | 13.96% |
| 10% | 16.13% | 20.61% |
| 20% | 29.16% | 37.13% |
| 30% | 43.49% | 53.96% |
| 50% | 65.27% | 69.86% |

These are **fractions of unfamiliar candidates**, not fractions of all Bangkok cells or of a real crew's workload. The rows show the capture curve; no single row is the recommended allocation. Ties at the cutoff receive fractional selection weight.

## Where the rank changes

Mean percentile uplift varies by prior reporting density: **+7.30, +1.98, −0.20, +9.39, +10.69 points** across the five ten-district density bands from lowest to highest. The weakest middle band shows why a citywide mean should not be treated as a uniform district benefit.

By distance to the nearest earlier report, the mean rank percentile is **0.685 → 0.689** for 271 sites 250–500 m away, **0.474 → 0.597** for 79 sites 500–750 m away, and **0.313 → 0.564** for 43 sites more than 750 m away. These are conditional ranks among unfamiliar cells; a high percentile does not establish that a feasible citywide crew queue would select a site.

## Decision boundary

The earlier 20% reserve comparison remains a **conditional experiment**, not an overall operational uplift. Its approximately 20 extra shortlisted report sites should not be projected into avoided flood losses. Previous source-district selection for **overall report-positive cell-day capture** chose **0% discovery reserve in all five folds**. That older result uses another outcome and cannot be numerically compared with these site-event capture percentages, but it makes the allocation tradeoff clear.

The next operational evaluation needs an observed crew-hour or stop-count budget. At that budget, score a full district queue including familiar and unfamiliar cells, count independently verified actionable sites and total sites, and record staff time. This ranking curve establishes that the CNN can order unfamiliar candidates better on retrospective report-site proxies; it does not establish the best share of work to assign to them.

## Reproduce

Run `.venv/bin/python flood_citywide_site_novel_rank_curve.py`. The script reads the previously trained five-fold CNN weights and source data, selects model settings on source districts using mean rank percentile, and writes [summary JSON](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_site_novel_rank_curve.json) and [event-level scores](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_site_novel_rank_curve_events.csv). Report-site construction and model fitting are documented in [the site-label study](FLOOD_CITYWIDE_PHYSICAL_SITE_TEST.md).
