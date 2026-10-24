# More open data for the Bang Kapi spatial flood-report model

Two data expansions are now tested: three earlier years of Bang Kapi reports and 2025 reports from five other Bangkok districts. Both models predict **whether a 250 m cell intersects a flood-complaint proxy on a day**. They do not predict verified standing water or depth. A cell without a report is a zero for this *report* target, not a verified dry cell.

## Data added

| Experiment | Additional data | Label rule |
|---|---|---|
| Earlier years | 937 direct-category Bang Kapi [Traffy Fondue](https://data.bangkok.go.th/) flood reports from [2022](https://data.bangkok.go.th/dataset/traffy-fondue-2565), [2023](https://data.bangkok.go.th/dataset/traffy-fondue-2566), and [2024](https://data.bangkok.go.th/dataset/traffy-fondue-2567), making 2,280 positive cell-days | Reports close in time and space connect through low [BMA LiDAR](https://cpudgiapp.bangkok.go.th/image/rest/services/DTM_LiDAR/ImageServer?f=pjson) terrain when possible; otherwise a 100 m report buffer. This is the same rule as the existing Bang Kapi CNN. |
| Other districts | 2025 [Traffy Fondue](https://data.bangkok.go.th/dataset/traffy-fondue-2568) reports from Chatuchak, Prawet, Suan Luang, Saphan Sung, and Wang Thonglang, alongside Bang Kapi | A uniform 100 m buffer around each direct-category report in all six districts. This isolates the effect of adding districts within this experiment. |

Both use [BMA district boundaries](https://data.bangkok.go.th/dataset/1e04f888-6287-41ce-aaa8-91f3bc6dae25), LiDAR-derived terrain, [BMA pump and gate locations](https://data.bangkok.go.th/dataset/water-station), and district-point [Open-Meteo historical hourly rain](https://open-meteo.com/en/docs/historical-weather-api). The earlier-years CNN also uses the existing [OpenStreetMap Thailand](https://download.geofabrik.de/asia/thailand.html) roads, buildings, waterways, and amenities. Neither CNN uses satellite imagery as an input. January 2025 OSM and available infrastructure features are applied retrospectively to 2022–2024; changes in roads, buildings, and pumps during those years are not represented. Rain is a historical gridded estimate at one point per district, not a measured gauge value in each cell.

The old Traffy archives are reduced to local dated/geocoded records and monthly cell activity. Free text, addresses, photo URLs and raw ticket IDs are not retained. The [archive audit](data/flood_bangkapi_surface/traffy_history_2022_2024_audit.json) records source row counts and category filtering.

## Held-out Bang Kapi results

Models train through 31 July 2025, select a checkpoint and calibrate on August–September 2025, and evaluate **1 October–31 December 2025**. All rows below rank the same 527 Bang Kapi cells by the number of report-positive days in that last period. Within each experiment the CNN architecture and inputs are held fixed while training geography or years change. The two experiments use different footprint rules and feature sets, so their values should not be read as a single controlled model ladder.

| Experiment and ranking | Spearman with observed cell counts | Pairwise agreement | Daily average precision | Predicted vs observed positive cell-days |
|---|---:|---:|---:|---:|
| Connected-footprint historical count, Jan–Sep 2025 | 0.490 | 77.7% | — | — |
| Connected-footprint per-cell tree, terrain/OSM/rain | 0.459 | 78.2% | 0.0198 | 209 vs 241 |
| Connected-footprint CNN, train on 2025 only | 0.368 | 72.6% | 0.0214 | 208 vs 241 |
| Connected-footprint CNN, add 2022–2024 | **0.492** | **80.5%** | **0.0385** | **105 vs 241** |
| 100 m footprint historical count, Jan–Sep 2025 | 0.490 | 77.7% | — | — |
| 100 m footprint per-cell tree, terrain/OSM/rain | 0.460 | 78.3% | 0.0198 | 209 vs 242 |
| 100 m footprint CNN, Bang Kapi training only | 0.492 | 80.5% | 0.0154 | 436 vs 242 |
| 100 m footprint CNN, six-district training | **0.513** | **82.1%** | **0.0151** | **1,150 vs 242** |

Earlier years raise the connected CNN's ranking and daily precision relative to its 2025-only counterpart and the per-cell tree. Its Spearman remains almost identical to the simple historical-count baseline. Adding five districts improves the controlled 100 m CNN's Spearman by **0.021** and pairwise agreement by **1.6 percentage points**. Against the 100 m per-cell tree, it wins geographic ranking but loses daily precision and calibration. The six-district model predicts nearly five times as many report-positive cell-days as observed. These are ranking experiments; neither model yields trustworthy flood probabilities.

The cross-district result uses 363 Bang Kapi training positive cell-days versus 2,265 across all six districts (January–July 2025). The earlier-years model uses 2,643 training positive cell-days, including older years. The different October–December totals, 241 and 242, come from the connected-area and uniform-buffer label rules respectively. The [multi-year metrics](artifacts/flood_bangkapi_surface/surface_connected_cnn_multiyear_metrics.json) and [cross-district metrics](artifacts/flood_bangkapi_surface/surface_crossdistrict_terrain_cnn_2025_metrics.json) give the full calibration and ranking calculations.

The multi-year model is also unsuitable for physical rain counterfactuals: holding intensity fixed, extending a scenario from two to four hours lowers the predicted report score in 12 cells at 4 mm/h and 17 cells at 8 mm/h. The CNN has no monotonic rain constraint, and these outputs were not checked against observed flood extent.

The static historical-count baseline cannot respond to a rain scenario. The per-cell tree **can**: it takes rain amount, peaks, and duration-related features, and constrains their individual effects to be non-decreasing. The CNN can also be fed altered rain, but its spatial structure alone does not make those outputs a validated simulation. This distinction matters when choosing a model for the intended rain-scenario use case.

Explore the [multi-year comparison map](artifacts/flood_bangkapi_surface/surface_connected_cnn_multiyear_comparison_map.html), [cross-district comparison map](artifacts/flood_bangkapi_surface/surface_crossdistrict_terrain_cnn_2025_comparison_map.html), [multi-year loss curve](artifacts/flood_bangkapi_surface/surface_connected_cnn_multiyear_loss_curve.png), and [cross-district cell table](artifacts/flood_bangkapi_surface/surface_crossdistrict_terrain_cnn_2025_cell_comparison.csv).

## Global-data check

The [AI for Good flood dataset](https://huggingface.co/datasets/ai-for-good-lab/ai4g-flood-dataset) has a radar-derived 2014–2024 recurrence tile covering Bang Kapi. In the district polygon, the cached tile has 83,008 pixels: 64,746 (78%) are value 1, which its documentation identifies as an exclusion mask that includes urban false-negative areas, and 18,262 are zero or nodata. There are **no value-2-or-higher flood detections** to add as positive Bang Kapi labels. Zeros here cannot supply verified dry observations. The counts are reproducible with [the radar audit](flood_bangkapi_global_radar_audit.py) and its [result](artifacts/flood_bangkapi_surface/ai4g_recurrence_bangkapi_audit.json). The [FloodSimBench dataset](https://huggingface.co/datasets/chrimerss/FloodSimBench) has useful 1 m DEM/rain-to-depth examples from simulated US city events, but its simulated water depths are a different target and geography; none were mixed into this complaint-label backtest.

## Interpretation and next evaluation

The selected October–December 2025 period has been inspected repeatedly in this project; treat these results as **exploratory**, not a fresh blind test. Other-district data have improved one Bang Kapi ranking, but this does not yet establish transfer to a new district. A credible next evaluation would freeze the inputs and model choices, then test an entirely held-out district or later storm period. Verified wet and dry observations, local rain, and pump operation data would be needed before fitting or calibrating a true surface-flood probability or answering "twice the rain for twice as long" as a validated counterfactual.

Rebuild with `.venv/bin/python flood_bangkapi_traffy_history.py`, `.venv/bin/python flood_bangkapi_spatial_cnn.py --use-history`, and `.venv/bin/python flood_bangkok_spatial_cnn_crossdistrict.py`. The first command fetches and streams the 2022–2024 source archives; the two model commands use the cached BMA and weather inputs.

The follow-on [multi-year, eleven-district experiment](FLOOD_BANGKOK_MULTIDISTRICT_MORE_YEARS.md) tests whether earlier reports from more Bangkok districts help Bang Kapi and a held-out Bang Khae transfer area.
