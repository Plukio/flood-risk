# Retrospective flood inspection ranking: 11 Bangkok districts

The expanded open-data backtest covers **11 districts and October–December 2023, 2024, and 2025**. At a budget of **25 cells per district**, the local model captures **241** later complaint-site cells, versus **251.3 expected** for the simple recent-report rule and **61.2 expected** for random selection. The model wins 13 district-years, loses 18, and ties 2 against recent reports. This does **not** support a claim that the model saves inspections. Both targeted rules are much better than random selection for this complaint proxy.

This is a retrospective comparison, not a field trial. The outcome is a geocoded direct-category [Traffy Fondue](https://data.bangkok.go.th/dataset/traffy-fondue-2568) flood complaint within a selected 250 m cell. It is **not verified inundation**. No BMA inspection queue, staff time, or verified dry-cell observations are available.

## How the backtest works

- On **30 September** of each test year, fit a separate local gradient-boosted tree for each district using daily complaint labels and features available through that date. The training label is a 100 m report buffer touching a 250 m cell. For ranking, freeze report history and average each cell's score over rainy October–December weather days from **earlier years only**. The score is a relative priority, not a calibrated flood probability.
- Compare equal budgets of **10, 25, 50, and 100 cells**. The simple comparator ranks cells by report-positive days over the preceding 12 months; another ranks by lowest LiDAR ground. Random selection is a reference. All methods are assessed against **October–December report points**, with distinct cells containing a point as the primary metric. A tie at the budget cutoff gets expected capture under random tie resolution.
- The tree uses LiDAR elevation and terrain, pump and gate proximity, mapped canal proximity, prior complaint history, season, and historical rain and antecedent wetness. Its fit uses no test-quarter rain or complaints. Static features include a **2025 OSM canal snapshot**, even in the 2023 and 2024 backtests; availability dates of the LiDAR and BMA infrastructure inventory are unproven. Those years are therefore retrospective stress checks rather than strict historical forecasts.

The test spans **6,312 distinct mapped cells**, with **1,368 district-year complaint-site cell outcomes** and **2,454 mapped report points** in the three test quarters combined. A fixed 25-cell budget covers different fractions of districts because their mapped cell counts range from 315 to 934.

## Overall fixed-budget results

Captured site cells are summed over 33 district-years; fractional captures are the expected result where the recent-report rule ties at the cutoff.

| Cells per district | Local tree | Recent reports | Random expectation | Tree wins / losses / ties vs recent reports |
|---:|---:|---:|---:|---:|
| 10 | 112 | 126.4 | 24.5 | 9 / 15 / 9 |
| **25** | **241** | **251.3** | **61.2** | **13 / 18 / 2** |
| 50 | 407 | 406.4 | 122.3 | 16 / 16 / 1 |
| 100 | 653 | 639.6 | 244.7 | 17 / 16 / 0 |

At 25 cells, the year totals for local tree versus recent reports are **64 vs 72.3** in 2023, **53 vs 53.2** in 2024, and **124 vs 125.8** in 2025. At 50 cells, the tree is effectively tied overall, though its year results differ: **99 vs 111.7**, **83 vs 81.3**, and **225 vs 213.4**. These totals are descriptive: the same districts recur across years, complaint volume differs sharply by year, and the actual BMA policy is unknown.

## Results by district and season at 25 cells

Each entry is **total site cells: tree capture / recent-report capture**. The latter can be fractional because of tied ranks. Every district-year, including model failures, is shown.

| District | Mapped cells | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|
| Bang Khen | 731 | 22: 9 / 7 | 7: 2 / 1 | 39: 7 / 9.5 |
| Bang Kapi | 527 | 39: 7 / 9.4 | 16: 3 / 3.6 | 63: 11 / 11.3 |
| Chatu Chak | 581 | 13: 2 / 1 | 17: 2 / 3.6 | 109: 12 / 10.4 |
| Pra Wet | 934 | 44: 5 / 6.8 | 42: 7 / 10.5 | 192: 17 / 15 |
| Suanluang | 448 | 39: 9 / 8.7 | 34: 8 / 9.5 | 75: 15 / 15.9 |
| Don Mueang | 652 | 20: 9 / 8.8 | 12: 4 / 4 | 71: 13 / 15 |
| Bang Khae | 840 | 57: 6 / 6.8 | 32: 6 / 3.3 | 65: 8 / 8.4 |
| Lak Si | 416 | 18: 7 / 7.5 | 6: 1 / 0.8 | 60: 17 / 14 |
| Saphan Sung | 514 | 25: 3 / 6.2 | 13: 6 / 2.7 | 61: 11 / 11 |
| Wang Thong Lang | 315 | 29: 3 / 4.4 | 13: 4 / 5.1 | 35: 8 / 7.8 |
| Bang Na | 354 | 29: 4 / 5.7 | 35: 10 / 9.1 | 36: 5 / 7.5 |

The strongest local-model gain at 25 cells is Saphan Sung 2024 (6 versus 2.7); the same district in 2023 reverses (3 versus 6.2). Pra Wet 2024 also favors recent reports (7 versus 10.5). The model's better performance than random selection is real for the complaint proxy, but **recent complaint history explains at least as much at a 25-cell budget**. A district-specific tree cannot currently be justified as a replacement for that simple ranking on capture alone.

The earlier [two-district backtest](FLOOD_BANGKOK_INSPECTION_BUDGET.md) also has a separate, small BMA road-site check: six approximately located Bang Kapi sites in Q4 2025. It is independent of Traffy complaints but too small to establish a gain. This expanded test uses Traffy outcomes consistently in all 11 districts; it does not turn unreported cells into verified dry cells.

The [BMA flood-depth audit](FLOOD_BANGKOK_DEPTH_AUDIT.md) checks the same districts and quarters. BMA reports contain road-flood heights, but lack event coordinates; the Traffy outcome points used here have no depth field. Depth therefore cannot be attached reliably to every ranked cell.

## Rebuild and inspect

Run `.venv/bin/python flood_bangkok_inspection_budget_expanded.py`. The script creates missing daily-cell caches from the cached public data and writes [all district-year metrics](artifacts/flood_bangkapi_surface/surface_inspection_budget_expanded_metrics.json) and [cell-level ranks and audit outcomes](artifacts/flood_bangkapi_surface/surface_inspection_budget_expanded_cell_ranks.csv). Outcome columns in the CSV are for audit after ranking; they are never model inputs for that test quarter. The 2023–2025 Traffy sources, [BMA LiDAR service](https://cpudgiapp.bangkok.go.th/image/rest/services/DTM_LiDAR/ImageServer?f=pjson), [2025 OSM](https://download.geofabrik.de/asia/thailand.html), and [Open-Meteo historical weather](https://open-meteo.com/en/docs/historical-weather-api) are open data.
