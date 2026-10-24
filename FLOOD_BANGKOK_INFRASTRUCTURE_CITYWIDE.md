# Do mapped drainage structures improve the citywide flood-complaint CNN?

We added seven static infrastructure channels to the 250 m CNN and backtested them against the saved 42-district baseline. The target is still a **100 m buffer around a direct Traffy flood complaint touching a cell-day**, not measured inundation. A cell without a complaint is zero for this proxy, not a verified dry cell.

## Open physical layers built

| Layer | Pretest open source | Usable coverage | Added channel(s) |
|---|---|---:|---|
| Canals | [OpenStreetMap Thailand, 1 Jan 2025](https://download.geofabrik.de/asia/thailand.html), `waterway=canal` | 3,896 ways in the citywide extraction bounds; mapped canal line in 9,277 of 28,309 valid cells | Log canal metres in cell; log distance to nearest mapped canal |
| Pump capacity | [BMA pump-station inventory](https://data.bangkok.go.th/dataset/pumpstation), CSV published 2023 | 118 unique geocoded stations with a usable reported capacity; 3,918 cells within 1 km | Log sum of reported capacities within 1 km |
| Retention ponds | [BMA pond GIS](https://data.bangkok.go.th/dataset/pond), 2021 | 25 geocoded points with storage volume; 3,893 cells within 2 km | Log nearby storage volume; log distance to nearest pond |
| Underpasses | [BMA underpass catalog](https://data.bangkok.go.th/dataset/tunnel), modified 2024 | 11 of 20 records have usable coordinates; 47 cells within 250 m, including 12 in the eight held-out districts | Indicator within 250 m; log distance to nearest mapped underpass |

The [source and feature audit](data/flood_bangkok_transfer/infrastructure_static_250m_v1/audit.json) records source URLs and hashes, validation counts, and district coverage. The [feature builder](flood_bangkok_infrastructure_features.py) saved aligned static layers for all 50 districts. The model's original 17 features and its outcome labels were unchanged. The BMA [drain-pipe inventory](https://data.bangkok.go.th/dataset/drain-pipe) was excluded: its 2,903 rows have pipe dimensions and road/from/to names, but no coordinates, and the accessible CSV was published in 2026. It cannot be placed reliably on the 2022–2025 cell grid.

These added channels are **asset proximity and density proxies**. They do not establish which cell drains into a particular canal, pipe, pond, or pump. Pond points are not pond footprints; pump capacity units need source confirmation; the asset catalogs do not give historical operating state. The 2025 canal snapshot and 2023–2024 catalogs are applied retrospectively to older training years.

## Matched backtest

The baseline and new variants fit CNN weights in the same **42 districts through 31 July 2025**, select a checkpoint by Bang Kapi **August–September 2025** daily AP, calibrate each district with its own August–September complaint labels, and test **October–December 2025** in all 50 districts. The same eight districts remain out of weight fitting. All variants use the same CNN architecture, optimizer, seed, 10,800 sampled district-days per epoch, and maximum 40 epochs with patience 18. The new model has 24 feature slots; group ablations set excluded channels to their normalized mean (zero) during fitting and evaluation. The original 17 channel normalization statistics match the saved baseline exactly.

Cell rank is mean **per-district Spearman** between each cell's expected and observed October–December complaint-positive days. Daily AP is mean per-district cell-day average precision. The last column sums expected complaint-positive cell-days across the eight held-out districts; **974** were observed.

| Inputs | Original 11 rank | Added 31 rank | Held-out 8 rank | Held-out 8 daily AP | Held-out expected / observed |
|---|---:|---:|---:|---:|---:|
| Jan–Sep 2025 cell-history rank | 0.428 | 0.304 | **0.318** | — | — |
| Original 17 channels | **0.435** | **0.320** | 0.309 | 0.0194 | **438 / 974** |
| 17 + **canal geometry** | 0.429 | 0.302 | 0.316 | **0.0230** | 365 / 974 |
| 17 + pump capacity, ponds, underpasses | 0.406 | 0.299 | 0.273 | 0.0206 | 361 / 974 |
| 17 + **all seven** new channels | 0.411 | 0.309 | 0.298 | 0.0189 | 387 / 974 |

Canal geometry gives a **small held-out gain** over the 17-channel CNN: mean Spearman rises **0.007** and daily AP rises **0.0036**. It improves cell rank in **four of eight** held-out districts and daily AP in **five of eight**. Its mean rank remains just below the simple history baseline, and its expected complaint count falls further below the observed count. The five point-asset channels, alone or combined with canals, reduce held-out geographic rank. Their sparse and uneven coverage and missing drainage connections limit interpretation; this test does not identify which individual asset channel causes the reduction.

The short [two-district canal pilot](FLOOD_BANGKOK_CANAL_FEATURES.md) also found mixed ranking and count effects. The current citywide comparison is a broader check, but the October–December 2025 period has been reused repeatedly, and the canal/assets split was examined after the all-seven result. Treat the small canal gain as exploratory. The model still ranks **complaints**, not actual wet cells or flood depths, and none of these inputs validates a doubled-rain flood simulation.

## Reproduce and inspect

Run `.venv/bin/python flood_bangkok_infrastructure_features.py`, then `.venv/bin/python flood_bangkok_spatial_cnn_infrastructure.py`, then `.venv/bin/python flood_bangkok_spatial_cnn_infrastructure_ablation.py`. The [all-seven metrics](artifacts/flood_bangkapi_surface/surface_crossdistrict_infrastructure_cnn_citywide_v1_metrics.json), [ablation metrics](artifacts/flood_bangkapi_surface/surface_crossdistrict_infrastructure_ablation_citywide_v1_metrics.json), [50-district comparison](artifacts/flood_bangkapi_surface/surface_crossdistrict_infrastructure_ablation_citywide_v1_district_comparison.csv), and [learning curves](artifacts/flood_bangkapi_surface/surface_crossdistrict_infrastructure_citywide_learning_curves.png) preserve the results and checkpoints.

For a stronger physical model, the next missing input is a **geocoded cell-to-outfall drainage graph** with pipe size and invert, canal connections, and timestamped gate/pump state. The current open pipe table and asset points cannot supply that graph directly.

**23 October 2026 correction:** this experiment inspected the BMA CSV, but a separate [public BMA ArcGIS drainage service](FLOOD_BANGKOK_DRAINAGE_GIS.md) has since been found with geocoded pipe, canal, pump, gate and tunnel layers. Its 2021 pipe geometry and partial invert fields can support a candidate graph. Street-inlet links, confirmed outfalls, and dated pump/gate operation are still missing. The model results above did not use the newly found GIS layer.
