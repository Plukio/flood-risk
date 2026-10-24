# Full district-cell flood-site capture curves

**24 October 2026.** This follow-up to the [unfamiliar-cell ranking analysis](FLOOD_SITE_QUOTA_FREE_RANKING.md) puts **all candidate 250 m district cells** on the x axis. Every district/day selects the same top fraction of its full candidate set. There is **no discovery-slot reserve**.

![Two-panel full-cell capture curves](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_site_all_cell_curves.png)

| Top share of all cells | All reported sites: history | All reported sites: CNN | Emerging sites: history | Emerging sites: CNN |
| ---: | ---: | ---: | ---: | ---: |
| 2% | 13.61% | 13.61% | 0.21% | 0.21% |
| 5% | 24.71% | 24.71% | 1.27% | 1.27% |
| **10%** | **38.32%** | **38.32%** | **3.50%** | **3.50%** |
| 20% | 55.75% | 55.58% | 11.17% | 11.38% |
| 30% | 67.87% | 67.89% | 21.66% | 22.61% |
| 50% | 83.64% | 83.10% | 44.37% | 46.60% |

The denominator is **2,326** Q4 2025 reported-site events in the left panel and **471** first-in-year sites, defined by no earlier report within 250 m during the preceding year, in the right panel. A site counts when its medoid cell enters the daily district shortlist. Boundary ties are split fractionally. These are reported-site proxies, not verified inundation.

## Scoring and selection

The reference orders every cell by the recent-plus-nearby report-history score. The model uses that same score and adds the trained site-CNN residual **only** for cells with no own-cell report in the prior 365 days; familiar cells retain their history score. A residual strength between zero and the trained strength is selected in January–September 2025 source districts to maximize the mean full-cell rank percentile of first-in-year sites. The CNN arm was also selected on source districts in the preceding quota-free analysis. Q4 2025 districts were excluded from fitting and these setting choices.

The selected residual strengths across the five geographic folds are **0.25, 0.25, 0.5, 0.5, and 0.5**. This is a single full-cell ranking, rather than two separate queues with a stipulated allocation. The source-chosen residual does not change either outcome at the tight top-10% all-cell screen. It yields small emerging-site gains at 20–50% of all cells, sometimes with lower total capture.

This explains the difference from the older **10.01% vs 5.68%** emerging-site result: that result forced **20% of a top-10% full queue** into unfamiliar cells. Here no such reserve is imposed, and the top-10% full queue shows **3.50% vs 3.50%** emerging capture. The 20% reserve was an illustrative allocation, not a validated staffing rule.

The earlier **65.7 vs 60.2** rank-percentile finding remains a valid *conditional* comparison among the 393 emerging sites located in zero-own-history cells. It does not by itself determine their priority relative to familiar cells. The present graph exposes that cross-group competition and shows no operational uplift at the 10% full-cell screen.

## Reproduce

Run `.venv/bin/python flood_citywide_site_all_cell_curves.py`. It writes the [curve metrics](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_site_all_cell_curves.json), [event-level selection weights](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_site_all_cell_curves_events.csv), and the PNG above. The Q4 site definition and model training are described in [the site-label study](FLOOD_CITYWIDE_PHYSICAL_SITE_TEST.md). Q4 and the 250 m definition were repeatedly examined during development, so this is an exploratory retrospective result.
