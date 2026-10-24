# Fixed-budget flood-site inspection test

**Expanded result:** the [11-district, three-season backtest](FLOOD_BANGKOK_INSPECTION_BACKTEST_EXPANDED.md) finds no consistent capture gain over the recent-report rule. At 25 cells, the local tree captures 241 later complaint-site cells versus 251.3 expected for recent reports across 33 district-years. The two-district results below are an earlier subset and should not be used alone to claim inspection savings.

This is a **retrospective selection test**, not a completed field inspection. We asked: if Bang Khae could inspect a fixed number of its 840 mapped 250 m cells just before October–December, how many cells with later flood complaints would be on that list? The open data do not contain BMA's actual inspection queue, verified wet/dry field checks, or inspection costs. Therefore the comparator called “recent reports” below is a transparent heuristic, **not BMA's current method**; future [Traffy Fondue 2024](https://data.bangkok.go.th/dataset/traffy-fondue-2567) and [2025](https://data.bangkok.go.th/dataset/traffy-fondue-2568) complaint points are problem-site proxies, not verified flooding.

## Decision and outcome rules

- **Decision date:** 30 September 2024 or 2025. Train each model on 2022 through that date. Freeze its local report history at the next day. Average each cell's score over rainy October–December days from **earlier years only**: 36 historical scenarios for the 2024 test and 48 for 2025. No test-quarter rain or reports enter the ranking. The score is relative priority, not a calibrated flood probability.
- **Budget:** 10, 25, 50, or 100 cells, treating each cell as one inspection slot; travel time and within-cell inspection effort are not modeled. A tie at the budget cutoff is evaluated by its expected capture under random tie resolution; the exported rank list uses row/column order as a stable display tie break.
- **Outcome:** a future direct-category Traffy report's geocoded **point** falls inside a selected cell in October–December. Multiple reports in one cell count as one problem-site cell for the primary metric; report points are a secondary metric. This stricter point rule avoids counting every neighbouring cell touched by the model's 100 m training buffer.
- **Comparators:** recent 12-month report-positive days per cell, lowest LiDAR ground, the baseline tree **refit through the decision date**, that tree with [surface cover](FLOOD_BANGKOK_NEW_FEATURES.md), and the Bang Kapi→Bang Khae transferred versions. Random selection provides a reference. The models use [BMA LiDAR](https://cpudgiapp.bangkok.go.th/image/rest/services/DTM_LiDAR/ImageServer?f=pjson), [2025 OSM](https://download.geofabrik.de/asia/thailand.html), [2021 ESA WorldCover](https://esa-worldcover.org/en), BMA infrastructure locations, and historical [Open-Meteo rain](https://open-meteo.com/en/docs/historical-weather-api). The 2024 run uses a 2025 OSM snapshot and is a retrospective stress check, not a strict as-of-2024 forecast. The infrastructure inventory's earlier availability is not established.

## Bang Khae results

| Budget | Selection rule | 2024 site cells / 32 | 2024 reports / 54 | 2025 site cells / 65 | 2025 reports / 105 |
|---:|---|---:|---:|---:|---:|
| 25 | Random expectation | 1.0 | 1.6 | 1.9 | 3.1 |
| 25 | Recent reports heuristic | 3.3 | 11.4 | **8.4** | **24.7** |
| 25 | Local baseline tree | **6** | **19** | 8 | 22 |
| 25 | Local tree + surface cover | **6** | 11 | 8 | 19 |
| 25 | Transferred tree + surface cover | 2 | 3 | 4 | 11 |
| 50 | Recent reports heuristic | 6.0 | 19.0 | 13.4 | **34.4** |
| 50 | Local baseline tree | **9** | **23** | 14 | 28 |
| 50 | Local tree + surface cover | **9** | 22 | **16** | 31 |
| 50 | Transferred tree + surface cover | 7 | 12 | 10.5 | 25.5 |

The local baseline tree catches more future complaint locations than random selection. Against the recent-report heuristic, its 25-cell result **wins in 2024 and loses slightly in 2025**. The heuristic needs 50 cells in 2024 to match the local tree's six captured site cells at 25, but needs only 24 cells in 2025 to match the tree's eight. That variation rules out a defensible claim of inspection slots saved. These are complaint-site proxies in any case, not measured field effort or money.

At 50 cells, the **local surface-cover tree** captures more distinct complaint-site cells than recent reports in both seasons: 9 versus 6 in 2024 and 16 versus 13.4 in 2025. Its report-point capture is mixed, and its 25-cell result does not improve on the existing local tree. The transferred surface-cover model fails to beat the simple recent-report rule at either budget in 2025. Its earlier gain in whole-district Spearman does not guarantee that the first cells are the most useful inspection queue.

## Separate BMA road-site check

The [BMA road-flood archive](https://data.bangkok.go.th/dataset/frd_dds) documents **six named Bang Kapi sites on eight October–December 2025 site-days** that can be attached to approximate cells using a **2026 site inventory**. At 25 cells, the local tree, local tree with surface cover, and recent-report heuristic each capture four of six sites. The transferred surface-cover tree has an expected capture of 3.8 under random tie resolution. These are independent positive road records, but the location anchoring is retrospective and six sites are too few to establish a reliable gain. They do not provide verified dry cells.

## Operational conclusion

The expanded backtest does **not** show a consistent capture gain over recent reports. This two-district test does **not** establish that the model currently saves inspections or that the transferred model should replace local prioritization. BMA's real savings require the actual queue, inspection cost, and field-verified wet/dry outcomes.

For a prospective test, freeze a 25-cell model queue and a 25-cell current-practice queue **before a storm season**, record why each site was selected and the staff time to inspect it, and use the same field/photo checklist at both sets plus a random sample of unselected cells. Measure newly verified drainage/flood problems found per inspection-hour and how many later flooded sites each queue covered. Keep the model fixed during the test; compare by storm and district before scaling.

The [prospective pilot kit](FLOOD_BANGKOK_PROSPECTIVE_PILOT.md) now supplies the queue-freeze command, current-practice and score templates, rain-event and inspection forms, a reviewed-field-results analyzer, and a clearly marked historical workflow rehearsal. It cannot create verified findings before field visits occur.

The [full budget metrics](artifacts/flood_bangkapi_surface/surface_inspection_budget_backtest_metrics.json), [all ranked cells](artifacts/flood_bangkapi_surface/surface_inspection_budget_backtest_cell_ranks.csv), and [historical 2025 Bang Khae top-25 queue](artifacts/flood_bangkapi_surface/surface_inspection_budget_backtest_bang_khae_2025_historical_top25.csv) allow review. The queue's future-outcome columns are **audit only**, not inputs to its ranks. Rebuild with `.venv/bin/python flood_bangkok_inspection_budget_backtest.py`.
