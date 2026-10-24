# Uphill built area during rain: citywide Bangkok backtest

Run 22 October 2026. The [reproducible script](flood_citywide_uphill_paved.py) builds candidate surface paths and learns one shared response to upstream built area and rain. The measured target is **Traffy report-positive 250 m cell-days**, not observed inundation or flood probability. An unreported cell is an unknown comparison, not verified dry.

## Paths and response

1. Mosaic the 50 cached [BMA DTM LiDAR exportImage](https://cpudgiapp.bangkok.go.th/image/rest/services/DTM_LiDAR/ImageServer?f=pjson) district rasters onto the existing 250 m Bangkok grid. The source exports are 512 × 512 per district with **3.74–45.13 m** source pixels, so this is not a uniform 10 m LiDAR grid. Terrain is available in 25,804 of 25,812 candidate cells.
2. Link each cell to its steepest lower adjacent cell if the 250 m cell-height difference is **0.05–5 m**. The graph does not stop at district borders: **1,159** of its 21,435 downhill links cross one. Follow each source cell's downstream path for **500–3,000 m**, downweighting travel length exponentially at a 1,500 m scale. These are terrain guesses; they omit drains, kerbs, walls, culverts, pumps and gates.
3. Multiply the [ESA WorldCover 2021](https://esa-worldcover.org/en) built fraction in each upstream source cell by its area and the source tile's retrospective [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api) full-day or peak-three-hour rain. Sum those contributions at downstream cells. WorldCover built cover is a **proxy**, not measured pavement or permeability. Rain is one district proxy per 20 km tile and can include hours **after** a complaint.
4. Fit a ridge-regularized **pairwise logistic residual** on top of the frozen recent + nearby complaint-history score. One global coefficient set responds to local built × rain, upstream built area, upstream built × daily rain, and upstream built × peak-three-hour rain. Positive sites are paired with sampled unreported cells on the same day. Rain-response coefficients are constrained nonnegative; this imposes a sensible directional response but does **not** validate rain scenarios.

The cross-district links have a median mapped drop of **0.326 m**, versus **0.297 m** for within-district links; this simple check found no large aggregate border jump, but it cannot certify individual paths.

The model uses **24,154** 2023–2024 positive cell-days and 193,232 sampled comparison pairs. The residual strength is selected on January–September 2025; October–December 2025 is scored afterward. The Q4 period had been inspected in earlier project work, so these comparisons are exploratory.

Controls use the same local feature and learned response form: a **500–3,000 m distance annulus** with globally matched total weight, and the same path links run in the **reverse direction**. The history queue receives no rain or built-cover adjustment.

## Result

Each percentage is the fraction of report-positive cell-days captured by the highest-ranked **10% of candidate cells each day**. Models in a row use the same days and candidate budget. The eight-district row is a separate fit: those districts' 2023–2024 labels and January–September 2025 labels are excluded from weight fitting and selection, while their earlier complaint counts remain available to the history score.

| Q4 2025 queue | Positive cell-days | History | Local built × rain | Distance control | Reverse-path control | **Uphill path** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All Bangkok; fit on all districts | 7,209 | 48.17% | 48.47% | 48.54% | 48.54% | 48.46% |
| Outside old training patches + 750 m; all-district fit | 3,605 | 46.08% | **46.50%** | 46.30% | 46.39% | 46.39% |
| Eight geographic holdout districts; fit on other 42 | 978 | 60.23% | 60.94% | **62.16%** | 61.13% | 60.84% |

The uphill path **has no demonstrated increment** over local cover × rain or the distance control. In the all-district fit its citywide difference from distance is **−0.08 percentage points**; a seven-day-block interval is **−0.37 to +0.35 points**. On eight geographic holdouts it is **−1.32 points** behind distance, with interval **−2.65 to +0.53**. The path's daily-rain coefficient after constrained fitting is small (**0.0148** on its transformed feature) and the peak-three-hour path coefficient reaches **zero**. An unconstrained diagnostic had a *negative* peak coefficient, which is why the reported model uses monotonic rain terms.

At this reach, terrain paths deliver upstream messages to only **3,240/25,812 cells (12.55%)**. Those cells contain **807/7,209** Q4 report-positive cell-days. Even inside this routed subset's own top-10% queue, uphill capture is **55.39%**, versus **55.51%** for local cover × rain and **55.64%** for distance. This suggests that sparse 250 m D8 paths, rather than only citywide dilution, limit this first feature.

## Interpretation and next physical improvement

The reusable pieces are the rule for making candidate paths, the feature units, and the shared response fit; each new city would rebuild its graph from its own terrain and cover. **Transfer to another city has not been tested.** This result also cannot be used as a flood map for twice the rain: neither open outcome provides reliable wet/dry extent, and the rain input is not aligned to the complaint hour.

For another attempt, improve the **paths and observations** before adding a larger GNN: derive flow at finer terrain resolution, resolve depressions/road crossings with credible drainage evidence, connect cells to mapped canals and outlets where justified, and use rain only through an explicit prediction cutoff. Keep the local, distance, reverse-direction and history controls. If better paths still fail those controls, there is no evidence that a learned long-range surface relation improves this report-ranking task.

The [all-district metrics](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_uphill_built_rain_ranker_monotone.json), [geographic holdout metrics](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_uphill_built_rain_ranker_geographic_holdout.json), and [unconstrained diagnostic](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_uphill_built_rain_ranker.json) preserve coefficients, selected residual strengths, district scores and paired intervals. Reproduce with `.venv/bin/python flood_citywide_uphill_paved.py` and `.venv/bin/python flood_citywide_uphill_paved.py --geographic-holdout`.
