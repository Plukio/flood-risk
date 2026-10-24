# Open-data flood-model feature expansion

We tested four feature groups on a 250 m Bang Kapi / Bang Khae flood-report model: finer LiDAR low points, surface cover, mapped canal-graph position, and spatially interpolated HII rain. The target remains **a 100 m Traffy flood-report buffer touching a cell on a day**, not verified inundation. A cell without a report is a sampled comparison cell, not proof that it was dry.

## Data and features

| Group | Input | Added information | Main limit |
|---|---|---|---|
| Terrain | [BMA LiDAR DTM](https://cpudgiapp.bangkok.go.th/image/rest/services/DTM_LiDAR/ImageServer?f=pjson), 10 m export | 10th percentile height, within-cell relief, local low-point depth and fraction, terrain-only downhill accumulation | Ground-only surface and boundary-truncated D8 flow cannot represent drains, walls, underpasses, pumps, or gates. |
| Surface cover | [ESA WorldCover 2021](https://esa-worldcover.org/en) and [OSM Thailand, 1 Jan 2025](https://download.geofabrik.de/asia/thailand.html) | Built, vegetated, and water class fractions; mapped building fraction and road length | Mapped class is not measured permeability; 2025 OSM is applied retrospectively to earlier years. |
| Canal graph | [OSM 2025 canals](https://download.geofabrik.de/asia/thailand.html) and [BMA pump/gate inventory](https://data.bangkok.go.th/dataset/water-station) | Mapped component length, along-line distance to a mapped gate/pump, attachment distance and reachability | Shared OSM nodes and nearest-node attachments do not prove hydraulic connectivity, flow direction, or operating state. |
| Spatial rain | [HII hourly rain archive](https://tiservice.hii.or.th/opendata/data_catalog/hourly_rain) and [station metadata](https://data.hii.or.th/dataset/40afcc66-a44c-4f64-8242-ab8d555b342d/resource/daeae663-3aea-4995-8a45-8f114e189a31/download/0all_stn_metadata.csv) | Inverse-distance interpolation of complete observed station-days within 20 km; daily amount, peak hour, nearest valid station distance, station count, missingness | It is gauge interpolation, not observed rain at each street. Timestamp time zone is unverified. |

The [static feature audit](data/flood_bangkok_transfer/new_static_250m_audit.json), [canal graph audit](data/flood_bangkok_transfer/canal_graph_250m_audit.json), and [HII quality audit](data/flood_bangkok_transfer/hii_rain_daily_2022_2025_audit.json) describe extraction. HII requires 24 valid hourly readings for each day, rejects invalid values and suspect all-zero wet-season months, and keeps missing readings missing. AIT001 supplies **no complete rain days**. In both districts, at least one usable gauge falls within 20 km for about **96% of cell-days**; two or more are usable for **92%** in Bang Kapi and **79%** in Bang Khae. Those figures do not imply street-scale rainfall accuracy. The base model retains [Open-Meteo historical rain](https://open-meteo.com/en/docs/historical-weather-api) at one coarse point per district, so the HII test measures the value of adding gauge-based information.

## Matched backtest

Every variant uses the same positive training cell-days, the same sampled 20% of comparison cell-days with inverse-probability weights, and the same gradient-boosted tree settings. Training is through July; evaluation is October–December. Report-history features are frozen at the test boundary, so later test reports cannot enter the inputs. **Cell Spearman** ranks the 840 Bang Khae cells by the sum of predicted report probability against observed report-positive days. **AP** ranks all 77,280 Bang Khae cell-days. Predicted count is the sum of probabilities; the observed count is 248 in 2025 and 156 in 2024. The 2025 block has been repeatedly inspected in earlier work; 2024 is a second exploratory season, not a pristine external validation set.

### Bang Kapi training → Bang Khae test

| Added group | 2025 cell Spearman | 2025 AP | 2025 predicted / 248 | 2024 cell Spearman | 2024 AP | 2024 predicted / 156 |
|---|---:|---:|---:|---:|---:|---:|
| None, controlled baseline | 0.218 | 0.0305 | 64 | 0.122 | 0.0105 | 88 |
| LiDAR low points | 0.219 | 0.0287 | 64 | 0.147 | 0.0120 | 90 |
| Surface cover | **0.276** | **0.0306** | 68 | **0.209** | **0.0218** | 90 |
| Canal graph | 0.240 | 0.0289 | 68 | 0.124 | 0.0113 | 82 |
| HII spatial rain | 0.265 | 0.0205 | 85 | 0.125 | 0.0096 | 115 |
| All four groups | **0.322** | 0.0253 | 90 | 0.187 | 0.0102 | 105 |

Surface cover is the most consistent addition for transfer: it improves geographic ranking in both seasons, and 2024 AP rises from 0.0105 to 0.0218. The all-feature model has the highest 2025 geographic rank, but lower AP than the baseline and a smaller 2024 improvement than cover alone. Every transferred variant **underpredicts the number of report-positive cell-days**. In 2025, even the all-feature model predicts 90 versus 248 observed. Its Brier score is 0.003186 versus 0.003192 for the baseline; that small change does not resolve the count error.

### Bang Khae training → Bang Khae test

| Added group | 2025 cell Spearman | 2025 AP | 2025 predicted / 248 | 2024 cell Spearman | 2024 AP |
|---|---:|---:|---:|---:|---:|
| None, controlled baseline | 0.300 | 0.0297 | 104 | 0.313 | 0.0201 |
| LiDAR low points | 0.318 | 0.0308 | 100 | 0.344 | 0.0189 |
| Surface cover | **0.353** | 0.0297 | 102 | 0.334 | 0.0181 |
| Canal graph | 0.307 | 0.0333 | 106 | 0.314 | **0.0211** |
| HII spatial rain | 0.301 | **0.0649** | 114 | 0.335 | 0.0146 |
| All four groups | 0.351 | 0.0491 | 110 | **0.358** | 0.0170 |

HII rain more than doubles local 2025 AP, yet lowers local 2024 AP and transfer AP in both seasons. This is not reliable evidence that gauge interpolation will improve an unseen district. Bang Khae's nearby BKK005 gauge has no complete October 2025 days, while other gauges fill gaps at varying distances. Quality-screened availability and a station-distance channel make that data shift visible to the model, but they cannot remove it.

## What this supports

Carry **surface cover** into a broader district-transfer test; keep the other groups as candidates, especially terrain for ranking and canal position for local day-level precision. The current tree can accept modified rain values for a scenario, but these backtests do **not** validate a “twice as much rain” flooding scenario: the label is reporting rather than water extent, drainage state is absent, and probabilities remain miscalibrated by event count. Before using it as a public flood-probability map, test more held-out districts/seasons and calibrate on target-area observations.

Full [metrics and feature definitions](artifacts/flood_bangkapi_surface/surface_new_feature_group_ablation_metrics.json) are reproducible with `.venv/bin/python flood_bangkok_new_feature_ablation.py` after the three extraction scripts `flood_bangkok_new_static_features.py`, `flood_bangkok_canal_graph_features.py`, and `flood_bangkok_hii_rain_extract.py`.

## Fixed-budget operational check

The later [fixed-budget inspection backtest](FLOOD_BANGKOK_INSPECTION_BUDGET.md) tests a different decision: a shortlist fixed on 30 September for the following October–December season, with no future rain or reports in its rank. At 25 Bang Khae cells, the baseline tree refit through September captured 6/32 future complaint-site cells in 2024 and 8/65 in 2025; adding surface cover captured 6/32 and 8/65. The recent-report heuristic captured an expected 3.3/32 and 8.4/65. Thus the whole-district Spearman gain above does not establish a gain at this inspection budget.
