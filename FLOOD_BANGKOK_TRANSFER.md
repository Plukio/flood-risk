# Bangkok flood-report transfer and rain scenarios

Run the open-data experiment from this directory:

```bash
.venv/bin/python flood_bangkok_transfer.py --rate-mm-h 4 --duration-h 2 --month 9
.venv/bin/python flood_bangkok_spatial_cv.py --district "Wang Thong Lang"
```

The first command builds the cached district, terrain, rainfall and report data.
The second tests geographic transfer and writes the scenario response for every
district using a model that has **never trained on that district's reports**.
The second command prints one district's out-of-district scenario after scoring
all 50. Changing the rate, duration, month or antecedent rain requires rerunning both
commands; the second reads scenario settings from `metrics.json`.

## What is predicted

The unit is a **district rain event**. A positive outcome means BMA recorded at
least one road-flood report in that district during or near the event. A missing
report is **unobserved**, not proof that roads were dry. The output is a model
score for *report occurrence*, not water depth, flooded area, or a calibrated
physical flood probability. It cannot rank individual streets across Bangkok.

The model combines event rain total, peak hourly rain, duration, rain in the
previous 24 hours, month, and district LiDAR elevation summaries. The rain +
terrain model is constrained so increasing the rain inputs cannot lower its
report score. This makes its scenario response internally consistent; it does
not validate the response against an actual doubled-rain event.

## Transfer test against observed outcomes

Five folds each hold out ten Bangkok districts. For each fold, the model learns
from the other 40 districts' events in 2021–2024 and predicts the held-out ten
in 2025. Thus all 50 districts receive a 2025 prediction from a model that did
not see their own reports or any 2025 outcomes during training. Predicted
report scores are summed over each district's observed 2025 rain events and
ranked against its count of events with BMA reports.

| 2025 district ranking method | Spearman agreement with observed report-event counts | Correct order among 1,111 comparable district pairs |
|---|---:|---:|
| Count of observed rain events alone | 0.068 | 52.9% |
| Rain model | 0.137 | 55.5% |
| Rain + terrain model | **0.211** | **58.5%** |

The rain + terrain model's district ranking is weak, despite beating these two
simple baselines. Among 13,995 held-out 2025 district events, 255 have a
matched BMA report. Event-level average precision is 0.052 for rain + terrain
and 0.050 for rain alone, compared with a report prevalence of 0.018. Terrain
does not consistently improve rank across the five folds. For example,
Saphan Sung is ranked fourth by predicted 2025 report events yet has no matched
report event. These results do **not** establish a transferable flood ranking.

The fold split excludes each held-out district's reports, but neighboring
districts may use the same weather grid cell. The 50 districts occupy only
22 distinct Open-Meteo grid cells. This tests transfer of district outcomes
within Bangkok, not transfer to a wholly unseen climate or another city.

## Rain scenarios

The default scenario is a constant 4 mm/hour for two hours in September, with
zero antecedent rain. The four inputs are:

| Scenario | Hourly rate | Duration | Event total |
|---|---:|---:|---:|
| Base | 4 mm/h | 2 h | 8 mm |
| Twice the intensity | 8 mm/h | 2 h | 16 mm |
| Twice the duration | 4 mm/h | 4 h | 16 mm |
| Both doubled | 8 mm/h | 4 h | 32 mm |

Twice the intensity and twice the duration have the same total rainfall but
different peak and duration inputs. For Wang Thong Lang, the out-of-district
model's *report scores* are 0.0097, 0.0146, 0.0169, and 0.0198 respectively.
Those numbers are **hypothetical model responses**, not measured chances that a
road will flood. All four default inputs are below the 99th percentile of the
training rain features in every fold; the CSV flags inputs beyond that range
for other settings. Staying in that range does not validate a counterfactual.

`artifacts/flood_bangkok_transfer/scenario_districts_spatial_oof.csv` contains
the four scenario scores and their differences for all 50 districts. Different
folds fit different models, so a citywide ordering of these scenario scores or
changes is exploratory. `district_ranking_spatial_oof_2025.csv` contains the
direct ranking versus 2025 observed reports. `spatial_cv_metrics.json` records
the overall and per-fold checks. The main script also writes
`scenario_districts.csv`, but that file uses a single model trained on 40
districts; use the spatial out-of-fold file when evaluating transfer to each
district.

## Open inputs and limits

| Input | Source | Role and main limit |
|---|---|---|
| BMA road-flood reports, 2021–2025 | [BMA open-data catalog](https://data.bangkok.go.th/dataset/frd_dds) | Outcome proxy. 1,298 of 1,435 rows match a rain-defined district event; reports are incomplete observations. |
| District boundaries | [BMA district shapefile](https://data.bangkok.go.th/dataset/1e04f888-6287-41ce-aaa8-91f3bc6dae25) | Geographic units. District averages hide street-scale depressions and drains. |
| LiDAR terrain | [BMA LiDAR ImageServer](https://cpudgiapp.bangkok.go.th/image/rest/services/DTM_LiDAR/ImageServer?f=pjson) | District elevation summaries. No pipe, pump, canal or tide dynamics in this model. |
| Hourly precipitation | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | Rain-event definitions and model features. Gridded historical rain misses some local storms. |

The report matching window includes reports up to three hours before a rain
event and six hours after it. It is a heuristic, and 137 reports do not match.
The model sees complete historical event rainfall, so this is a retrospective
backtest and scenario exploration, not a lead-time warning system.

## What would make street-level scenarios credible

Geocode the BMA report location text to road segments and manually audit those
matches. Collect open, time-stamped wet and dry observations from available
imagery or road sensors; the public CCTV catalog currently provides locations
but no historical frames. Add local high-frequency rain and observed
pump/canal/tide states if they become accessible. Then hold out whole districts
and storms, rank street segments against observed flooded segments, and test
intensity and duration changes on separate historical storms. A U-Net needs
many reliable event-aligned flood masks; the present data do not provide them.
