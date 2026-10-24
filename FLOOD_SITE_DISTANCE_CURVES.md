# Emerging-site capture by distance from earlier reports

**24 October 2026.** This extends the [quota-free unfamiliar-cell ranking test](FLOOD_SITE_QUOTA_FREE_RANKING.md). The x axis is the share of **all unfamiliar candidate cells** reviewed in each district/day. “Unfamiliar” means no report in that 250 m square during the preceding 365 days. There is no reserved share of the *full* district queue. Each panel ranks the **same candidate pool**; only the later reported-site outcomes are split by distance to the nearest report on an earlier day.

![Emerging-site capture curves by prior-report distance](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_site_distance_rank_curves.png)

The site outcome is a same-day Traffy point-cluster medoid with no earlier report within 250 m in the preceding year. Of 471 Q4 2025 emerging site events, **393** are located in zero-own-history cells and enter this test. Distance bands partition those 393 events.

| Nearest earlier report | Sites | Mean rank gain, CNN − history | 10% unfamiliar cells: history → CNN capture | 30% unfamiliar cells: history → CNN capture |
| --- | ---: | ---: | ---: | ---: |
| 250–500 m | 271 | +0.5 percentile points | 22.3% → 24.7% | 54.6% → 59.8% |
| 500–750 m | 79 | +12.3 points | 3.8% → 13.9% | 26.6% → 43.0% |
| >750 m | 43 | +25.2 points | 0.0% → 7.0% | 4.7% → 37.2% |

At the 30% screen, the gain grows with distance: **+5.2, +16.5, then +32.6 percentage points** across the three bands. The cutoff curves are not strictly monotone at every screen size: at 10%, the 500–750 m group gains more than the >750 m group. Mean rank is a cutoff-free check and rises by **+0.5, +12.3, then +25.2 percentile points**. Paired district bootstrap intervals for those mean-rank changes are **−3.2 to +3.6**, **+6.4 to +18.6**, and **+18.7 to +32.4 points**, respectively. The intervals do not account for repeated Q4 model-development inspection.

This pattern is consistent with the CNN helping most where nearby history is weakest. It does **not** prove that the CNN has learned physical water flow: terrain, built cover, reporting behavior, and training-label patterns can all affect the ranking. The >750 m band contains only 43 events. Crucially, [the full all-cell queue](FLOOD_SITE_ALL_CELL_CURVES.md) still shows **no emerging-site improvement at its top-10% cutoff**. Better ordering among unfamiliar cells does not establish a feasible inspection allocation.

## Reproduce

Run `.venv/bin/python flood_citywide_site_distance_curves.py`. It reads the frozen [event-level unfamiliar-cell scores](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_site_novel_rank_curve_events.csv) and writes the [subgroup metrics](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_site_distance_rank_curves.json) and chart above. It does not refit or retune the model on Q4 site outcomes.
