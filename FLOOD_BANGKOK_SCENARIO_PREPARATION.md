# Bangkok rain scenarios and preparation exercise

This is a **rain-input stress test** of the saved canal-feature CNN and CNN–GNN across all 28,309 valid 250 m cells in 50 districts. It is not a hydraulic flood simulation. Both models were trained to predict **Traffy flood-complaint occurrence**; BMA road-flood records supply a separate positive-site check, not complete wet/dry ground truth.

The scene freezes static terrain, infrastructure, season, and complaint history as of **1 October 2025**. Rainfall is one daily, district-wide summary based on a single Open-Meteo historical point per district. We replace its seven rain channels consistently from hypothetical hourly rates, durations, and prior totals. No later complaint labels enter the scenario inputs. The models' October–December 2025 outcome period has been examined repeatedly, so this is an exploratory retrospective exercise rather than a fresh prospective forecast.

## What we can vary

| Input | Available in these models | Meaning |
|---|---|---|
| Today's rain total, peak 1 h and 3 h, wet-hour count | Yes | Compare stronger rain with longer rain **within one day**. |
| Previous 24 h and 72 h rain | Yes | Stress-test antecedent wetness, without modeling drain storage. |
| Terrain, mapped canals, pump/pond/underpass proximity | Static only | Changes *where* the model places complaint risk, but does not simulate flow or asset operation. |
| Hour-by-hour water movement, depth, flooded area, time-to-drain | No | Requires a connected drainage/flow model and measured event outcomes. |
| Gate/pump failure, tide, blocked pipes, cross-district drainage | No | The open data and current graph lack reliable time-varying state and pipe-to-outfall links. |

The training rain archive spans **54,936 district-days** across the 42 model-fitting districts through July 2025. The largest training daily total is **76.7 mm**. The support counts below are the number of training district-days with *all seven rain summaries at least as high* as the scenario, not exact matches.

| Synthetic scene | Rain today | Prior 24/72 h | Training support count | Canal CNN score index | CNN–GNN score index |
|---|---:|---:|---:|---:|---:|
| Dry | 0 | 0 / 0 | 54,936 | 0.19 | 0.27 |
| Reference | 20 mm in 2 h | 0 / 0 | 502 | **1.00** | **1.00** |
| Twice the intensity | 40 mm in 2 h | 0 / 0 | 3 | 1.24 | 1.05 |
| Twice the duration | 40 mm in 4 h | 0 / 0 | 35 | 1.99 | 1.66 |
| Both doubled | 80 mm in 4 h | 0 / 0 | **0: beyond training maximum** | 2.43 | 1.73 |
| Wet beforehand | 20 mm in 2 h | 20 / 40 mm | 21 | 1.37 | 2.97 |

**Score index** is the sum of each model's locally intercept-adjusted complaint-proxy scores divided by its own reference-scene sum. It is **not a multiplier for flooded cells, flood area, water depth, or affected people**. The models have underpredicted held-out complaint-positive cell-days, and their score levels are not interchangeable.

The models disagree sharply about antecedent rain. With wetter prior days, the canal CNN *reduces* the score in **55% of cells** even as its citywide index rises; the CNN–GNN almost triples its index. With twice the rain intensity, the CNN–GNN reduces scores in **19% of cells**. These are valid descriptions of learned model behaviour, but they are not physically monotone water-flow predictions. The 80 mm scene is out of training support and should only be used to expose failure modes.

Spatial ranking barely changes under the within-day rain scenes: mean per-district Spearman against the 20 mm reference stays about **0.99** for both 40 mm scenes. At a fixed ten-cell inspection budget, about **nine of ten** reference cells remain on each model's 40 mm shortlist on average. With wet antecedent rain, only **7.6** (canal CNN) or **6.6** (CNN–GNN) remain. Thus these models mostly rescale familiar hotspots; they do not identify a credible newly flooded area when rain doubles.

## Preparation exercise

The [citywide review queue](artifacts/flood_bangkapi_surface/surface_citywide_rain_scenario_stress_v1_preparation_queue.csv) contains ten cells per district: up to five with the most January–September 2025 complaint-positive days, then enough cells to fill ten by agreement between the two models across four scenarios with at least some training support. The 80 mm extrapolation is excluded from selection. The queue gives a coordinate and a **suggested field check**, such as street-drain inspection, canal-side inlet review, pump-state verification, or underpass readiness. These are checks to verify on site, not assertions that an asset drains a cell. The [interactive Bang Kapi map](artifacts/flood_bangkapi_surface/surface_citywide_rain_scenario_stress_v1_bang_kapi_map.html) lets users switch model and rain scene and shows the ten review cells.

The queue has **500 cells** total, split 250 selected by complaint history and 250 by model consensus. As a retrospective independent check, it intersects only **9 of 89** conservatively located BMA Q4 2025 positive road-flood site-days, and **0 of five** in the geographic holdouts. Those BMA event coordinates are approximate matches to a 2026 site inventory. This small positive-only check does **not** validate the queue for operations; official BMA road reports must remain a separate source for reviewing known flood sites. The [queue audit](artifacts/flood_bangkapi_surface/surface_citywide_rain_scenario_stress_v1_preparation_audit.json) records the counts and rank-overlap checks.

For preparation now, the defensible use is:

1. **Verify known BMA road-flood locations** with district engineers; the public archive gives road/area text and depth, not precise event coordinates.
2. **Use the review queue for pre-storm field checks** of drains, canal-side inlets, actual connections, pumps, and underpasses. Keep the official-site list alongside it rather than treating either as complete.
3. **Update priorities during rain** with actual gauge totals, canal levels, CCTV/field observations, pump and gate status, and new BMA/Traffy reports. Record checked cells that remain dry as well as flooded cells.
4. **Do not turn the score index into closure or evacuation thresholds.** Those require time-matched depth/extent observations and a connected hydraulic network, followed by backtests on unseen storms and districts.

## Artifacts

The [scenario script](flood_bangkok_scenario_stress.py) produces [assumptions and training support](artifacts/flood_bangkapi_surface/surface_citywide_rain_scenario_stress_v1_metrics.json), [50-district scenario summary](artifacts/flood_bangkapi_surface/surface_citywide_rain_scenario_stress_v1_district_summary.csv), [all cell scores](artifacts/flood_bangkapi_surface/surface_citywide_rain_scenario_stress_v1_cells.csv), and the [score-index chart](artifacts/flood_bangkapi_surface/surface_citywide_rain_scenario_stress_v1_score_index.png). The [queue and map script](flood_bangkok_scenario_prepare.py) creates the review queue and Bang Kapi map. Reproduce with `.venv/bin/python flood_bangkok_scenario_stress.py` then `.venv/bin/python flood_bangkok_scenario_prepare.py`.
