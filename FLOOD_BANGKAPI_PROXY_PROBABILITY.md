# Bang Kapi report-footprint probability map

[Open the connected-footprint probability map](artifacts/flood_bangkapi_surface/surface_connected_probability_map.html) or [inspect the inferred footprints](artifacts/flood_bangkapi_surface/connected_report_footprint_map.html). The map covers **527 Bang Kapi cells of 250 m**, including off-road land. Its percentages estimate whether an **inferred flood-report footprint touches a cell on a day** under a chosen rain scenario. A cell-day with no footprint is treated as dry **only for this working proxy**. These percentages are not the probability of independently observed inundation.

## Inferring the footprint from connected reports

The [2025 BMA Traffy Fondue archive](https://data.bangkok.go.th/dataset/traffy-fondue-2568) has **329** direct flood-category reports geocoded within Bang Kapi. Two reports can join when they are at most **3 hours** and **300 m** apart, their ground elevations differ by at most **0.75 m**, and a route through the 10 m [BMA LiDAR terrain](https://cpudgiapp.bangkok.go.th/image/rest/services/DTM_LiDAR/ImageServer?f=pjson) stays no more than **0.25 m above the higher report point**. The path is searched in a **150 m wide corridor**. A connected group can span no more than 6 hours. These are exploratory parameters, not measured flood hydraulics.

For groups with at least two reports spread at least 50 m apart, the candidate area is low terrain connected to their points, clipped to the reports' convex hull plus 100 m and the district boundary. Otherwise each report gets a **100 m radius fallback**. The 100 m is a spatial proxy, not a measured water edge or a grid size. The [footprint GeoJSON](artifacts/flood_bangkapi_surface/connected_report_footprints_2025.geojson) and [method audit](artifacts/flood_bangkapi_surface/connected_report_footprints_audit.json) make the rule inspectable.

This yields **18 terrain-connected groups covering 54 reports** and **259 fallback groups covering 275 reports**. Their footprints create **811 positive cell-days** in **267 cells** during 2025. The former [100 m-only label rule](artifacts/flood_bangkapi_surface/surface_proxy_probability_map.html) creates 812 positive cell-days; only **five cell-day memberships differ** at 250 m resolution (two added and three removed). [Positive connected cell-days](artifacts/flood_bangkapi_surface/surface_connected_positive_cell_days_2025.csv) contain no comments, addresses or photo URLs. Terrain connectivity is a heuristic: it does not account for buried drains, pumps, walls, flood depth or whether adjacent reports describe the same event.

## Model and held-out check

The model uses LiDAR elevation, [OpenStreetMap](https://download.geofabrik.de/asia/thailand.html) roads/buildings/waterways/amenities, nearby [BMA pump and gate locations](https://data.bangkok.go.th/dataset/water-station), earlier report counts and complaint activity, plus hourly [Open-Meteo historical precipitation](https://open-meteo.com/en/docs/historical-weather-api). One Open-Meteo grid point supplies rain to every cell, so local storms may be missed. Sentinel-2 was excluded because its earlier ablation worsened geographic rank.

Training uses January–July 2025. An intercept-only correction uses August–September. October–December is held out, with report history frozen at 30 September. The correction keeps the cell ordering and monotonic rain response. This is a retrospective backtest using an archive published after 2025.

| October–December 2025, 48,484 cell-days | Connected-footprint result |
|---|---:|
| Positive proxy cell-days observed | 241 |
| Positive proxy cell-days predicted | 208.9 |
| Cell-count Spearman rank | 0.459 |
| Correct order of comparable cell pairs | 78.2% |
| Daily average precision | 0.0198 versus proxy prevalence 0.0050 |
| Brier score | 0.004922 versus constant-prevalence 0.004946 |

### Historical cell baseline

For a simple benchmark, rank each cell by its **number of positive report-footprint days from January–September 2025**. Compare that rank with the number of positive days observed in October–December. This baseline uses only information available before the held-out period. The model score is its expected positive days across October–December using those days' weather. [Compare the three cell maps](artifacts/flood_bangkapi_surface/surface_connected_historical_comparison_map.html): historical baseline, model prediction and held-out outcome. The [comparison CSV](artifacts/flood_bangkapi_surface/surface_connected_historical_comparison.csv) and [GeoJSON](artifacts/flood_bangkapi_surface/surface_connected_historical_comparison.geojson) contain all 527 cells.

| Label rule | Historical count Spearman | Model Spearman | Historical pairwise agreement | Model pairwise agreement |
|---|---:|---:|---:|---:|
| Connected footprint | **0.490** | 0.459 | 77.7% | **78.2%** |
| 100 m radius only | **0.490** | 0.460 | 77.7% | **78.3%** |

The historical count beats the model on **Spearman**, while the model has slightly higher pairwise order agreement. The terrain-connected label rule changes the mapped evidence, but this 250 m backtest shows **no ranking gain** over the 100 m rule. All these ranks concern the report-footprint proxy, not verified flood extent.

The [full metrics](artifacts/flood_bangkapi_surface/surface_connected_probability_metrics.json) show weak rain-specific calibration. October had 144 positive cell-days versus 115.5 predicted; November 94 versus 62.3; December 3 versus 31.1. On days with **1–10 mm** at the Open-Meteo point, 165 positives were observed versus 63.3 predicted. On days with **at least 10 mm**, 48 were observed versus 82.1 predicted. These discrepancies mean a particular printed percentage should not be used as a reliable flood probability.

The October scenario selector compares 4 mm/h for 2 h, 8 mm/h for 2 h, 4 mm/h for 4 h, and 8 mm/h for 4 h, with zero rain in the preceding 72 hours. Doubling intensity or duration alone changes the rounded probability in only **217 of 527 cells**; doubling both changes all cells. These are responses of a report-proxy model, not a validated physical flood simulation.

Download the [scenario CSV](artifacts/flood_bangkapi_surface/surface_connected_probability_scenarios.csv) or [GeoJSON](artifacts/flood_bangkapi_surface/surface_connected_probability_scenarios.geojson). `proxy_flood_probability_*` refers to the connected-report target; `flood_probability` remains empty for independently observed inundation. The separate [open-data evidence audit](FLOOD_BANGKAPI_SURFACE.md) explains that gap.

Rebuild with `.venv/bin/python flood_bangkapi_proxy_probability.py` (connected rule by default). Use `--mode buffer` to rebuild the 100 m-only comparison.

A [2D CNN experiment](FLOOD_BANGKAPI_SPATIAL_CNN.md) adds neighboring cells' features to the model and compares its held-out ranking with the historical baseline and this cell-feature model.
