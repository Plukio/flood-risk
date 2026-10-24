# Drainage state, antecedent wetness, and reporting activity

The new open-data backtests **do not establish a general fix for flood-report count bias or cell ranking**. They test three ideas raised after the [river-level audit](FLOOD_BANGKOK_WATER_LEVEL_BIAS.md): longer antecedent rain, a lagged reporting-activity proxy, and interaction between river level and mapped drainage proximity. The target remains a **100 m direct Traffy flood-report buffer touching a 250 m cell-day**, not independently measured inundation. All 2025 October–December results are exploratory because this period has been inspected repeatedly.

## What is measurable today

| Candidate | Open source and usable timing | Status |
| --- | --- | --- |
| Rain in the preceding 4–7 and 8–14 days | Existing [Open-Meteo hourly historical rain](https://open-meteo.com/en/docs/historical-weather-api), summarized from Bangkok local-time days before the target. The base model already uses the preceding 72 hours. | Tested in both districts, 2023–2025. This remains one gridded point per district. |
| Non-flood reporting activity | Public [BMA Traffy annual archives](https://data.bangkok.go.th/dataset/traffy-fondue-2568), streamed into **district-month counts** after excluding flood-category text from `type` and `problemtype_tag`. Only the *previous* month, normalized by its days, enters the model. | Tested. Available extracted months per district: 9–10 in 2022, 7 in 2023, and all 12 in 2024 and 2025. Missing months are missing values, never zero reports. |
| River boundary condition | [BMA Rama VIII daily river maximum](https://data.bangkok.go.th/dataset/river_rama), from the preceding day. | Tested again alongside wetness/activity. It is a river proxy, not a local outfall level or pure tide. |
| Inside and outside level at the **same gate and hour** | [BMA inner](https://data.bangkok.go.th/dataset/water_level_inner_daily) and [outer](https://data.bangkok.go.th/dataset/water_level_outer_daily) canal CSVs start in **2026** and contain daily maxima, not hourly values. Their station lists do not give an identified inside/outside pair at one gate. The [five-minute catalog](https://data.bangkok.go.th/dataset/flood) exposes a dictionary but no downloadable historical measurements. | **Cannot calculate a time-matched hydraulic head for 2022–2025.** Subtracting different daily maxima would be misleading. |
| Gate opening and operating pump rate | The [BMA water-station inventory](https://data.bangkok.go.th/dataset/water-station) contains location and physical equipment fields for 438 sites, but no timestamped operation field. The available [pump](https://data.bangkok.go.th/dataset/pumping-station) and [gate](https://data.bangkok.go.th/dataset/floodgate) catalogs also describe sites rather than historical operating state. | Static proximity can be tested; actual operation cannot. |

The [aggregate activity extract](data/flood_bangkok_transfer/traffy_nonflood_district_month_2022_2025.csv) and [source audit](data/flood_bangkok_transfer/traffy_nonflood_district_month_2022_2025_audit.json) retain no report text, coordinates, image, or identifier. Rebuild with `.venv/bin/python flood_bangkok_reporting_activity.py`. The source-assigned district is used; this differs from the polygon-checked flood labels. The activity series is a measure of **reporting behavior**, not a measured probability that a flood gets reported.

## Daily count comparison

The matched Poisson models train through July and receive the same August–September level adjustment before each October–December test. The 2025 table highlights the tension between matching a three-month total and predicting individual days:

| Inputs beyond month and existing rain | Bang Kapi predicted / 242 observed | Bang Kapi daily MAE | Bang Khae predicted / 248 observed | Bang Khae daily MAE |
| --- | ---: | ---: | ---: | ---: |
| None | 136.5 | **2.798** | 186.6 | **2.914** |
| Rain 4–14 days earlier | 162.1 | 3.141 | **215.7** | 3.013 |
| Previous-month non-flood reports | 164.2 | 3.018 | 302.3 | 3.420 |
| Longer wetness + activity | 178.4 | 3.223 | 355.9 | 3.686 |
| Longer wetness + activity + prior river | **195.4** | 2.847 | **242.4** | 3.554 |

The combined model's Bang Khae total, 242.4 versus 248, is **not** good daily calibration: its daily MAE rises from 2.914 to 3.554. Across the six district-year test blocks, no added-feature variant beats the rain-only model on daily MAE consistently. The earlier 2023–2024 blocks and daily Poisson deviance are in the [count metrics](artifacts/flood_bangkapi_surface/surface_shared_state_count_backtest_metrics.json); the [2025 daily table](artifacts/flood_bangkapi_surface/surface_shared_state_count_backtest_2025_daily.csv) permits date-level inspection. Rebuild with `.venv/bin/python flood_bangkok_shared_state_backtest.py`.

The count model's November and December activity inputs come from the prior October and November **non-flood** reports. This is a sequential backtest using information that would have been available by that month, not a single forecast issued before October. Full-day observed rain also makes it retrospective. The 2022/2023 activity gaps and a 2025 reporting-rate shift limit causal interpretation.

## Does water state move the cell ranking?

A separate matched cell-level tree comparison uses local elevation, historical report days, daily rain, longer wetness, mapped OSM canal length/distance, and BMA gate/pump distance. It then adds (1) yesterday's river level, and (2) explicit river × canal-and-gate/pump proximity terms. All variants use the same 20% negative-cell sample with inverse sampling weights and train through July 2025. October–December test history stays frozen after September. The **proximity terms are not a verified canal network or physical connection**.

| Bang Khae 2025 cell ranking | Cell Spearman | Cell-day average precision | Predicted / 248 positive cell-days |
| --- | ---: | ---: | ---: |
| Local model, no river | 0.300 | 0.02970 | 104 |
| Local model, river level | **0.303** | **0.03466** | 101 |
| Local model, river × drainage proximity | 0.295 | 0.03344 | 97 |
| Bang Kapi → Bang Khae, no river | 0.218 | **0.03051** | 64 |
| Bang Kapi → Bang Khae, river level | **0.223** | 0.02756 | 69 |
| Bang Kapi → Bang Khae, river × drainage proximity | 0.217 | 0.02526 | 72 |

Adding an actual prior river reading helps one local metric slightly; **the explicit drainage interactions do not improve local cell rank or transfer**. The static inventory also shifts: the median nearest-gate distance is about **1.54 km in Bang Kapi** and **3.63 km in Bang Khae**, so a shared proximity formula changes scale across areas. This simple tree is a diagnostic comparison, not a retrained version of the existing CNN. See the [spatial metrics](artifacts/flood_bangkapi_surface/surface_spatial_water_state_backtest_metrics.json) and `.venv/bin/python flood_bangkok_spatial_water_state.py`.

## Data contract for a real drainage-state feature

For each connected outlet, we would need: gate ID and mapped upstream cells; same-datum **inside and outside level with measurement timestamps**; gate open fraction; pump running capacity; and quality/staleness flags. Join readings from the same time window **before** the rain/report prediction cutoff. Then use `inside level − outside level` as a directional head feature, with gate and pump operation as separate inputs. Compare a model with those inputs against rain-only and historical-report baselines on new districts and storms. No current artifact asserts this head or operating state, because those measurements are absent from the open historical files.
