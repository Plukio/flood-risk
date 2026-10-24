# Does the Lat Krabang flow map help find Traffy report sites?

**24 October 2026. Short answer: the tested transfer does not improve the report queue.** The [terrain-crossing scenario map](FLOOD_LAT_KRABANG_EDGE_FLOW_PILOT.md) ranks GISTDA mapped water better than its ordinary and reversed flow controls for one selected rain scenario, but its score alone trails the frozen masked CNN on Traffy sites. January–September report validation chooses **no flow blend**.

## Matched report-site check

The test uses **999 fully terrain-covered 250 m cells** assigned to Lat Krabang. A saved [five-fold masked CNN](FLOOD_CITYWIDE_CONNECTED_SCORE_SMOOTHER.md) was trained without Lat Krabang or the other districts in its fold. Its existing report head and static normalization are frozen. The new input is each cell's areal-resampled share of simulated 10 m pixels over **10 cm** deep, tested for controlled rain of **20 mm/2 h, 60 mm/2 h, and 120 mm/6 h**. All three scenarios, ordinary-flow and reversed-flow controls, and small CNN blends are evaluated. The scenario selected using **2026 GISTDA** is **not** imported as a choice for this earlier 2025 report test.

At each date, the queue selects the top **10% of these 999 cells**. It reserves **20% of the inspection slots** for cells with no own-cell report in the preceding 365 days, matching the earlier novel-site experiment. The other slots use the fixed recent-plus-nearby history score. January–September 2025 selects a blend; **Q4 2025** measures it. Q4 has been examined in related experiments, so this is exploratory. There is no actual BMA queue or field finding here.

| Novel-site ranking on Q4 2025 | Captured first-in-prior-year report cell-days | Capture of 35 positive cell-days | Overall report capture of 78 positive cell-days |
| --- | ---: | ---: | ---: |
| Frozen masked CNN | 3.98 expected | **11.37%** | **56.38%** |
| Candidate-pass flow, 20 mm/2 h | 1 | 2.86% | 52.56% |
| Candidate-pass flow, 60 mm/2 h | 2 | 5.71% | 53.85% |
| Candidate-pass flow, 120 mm/6 h | 2 | 5.71% | 53.85% |
| Candidate-pass flow 120 mm/6 h + 25% blend into CNN rank | 0 | 0% | 51.28% |

The source-period novel capture was **10.17% for the CNN** (6 of 59) and **0% for each candidate-pass flow score**. The source-selected choice is therefore the **unchanged CNN**. One ordinary-flow blend reaches **11.43%** on Q4, effectively the same four captured sites as the CNN's **11.37%** after tie allocation; it was not selected on the source period and provides no credible gain. The direct history-only full queue captures **55.13%** of all 78 Q4 report-positive cell-days, versus the CNN queue's **56.38%**.

This local **11.37%** CNN figure cannot be compared directly with the earlier **~14% citywide, five-fold** novel-site result: the candidate geography and the positive-site denominator differ.

## Why the transfer is weak here

The observed targets occupy different surfaces. In the 20 October 2026 GISTDA pilot, **88.17%** of mapped 10 m pixels fall in WorldCover cropland. Among the 78 Q4 2025 Traffy report-positive cell-days in this Lat Krabang window, the mean cropland fraction is only **7.54%** and mean built fraction **49.92%**. Only **5 of 78** report-positive cell-days have majority cropland; **35 of 78** have majority built cover. A score that ranks broad mapped water in fields can miss where people report flooded roads or property.

The flow map also uses spatially uniform, controlled rain and a closed surface with **no drains, pumps, infiltration, tide, barriers, or time-matched operations**. Its 250 m storage cell assumes instant mixing inside the cell. Its modeled depth is not observed depth. The BMA DTM acquisition date has not been confirmed for a strict 2025 forecast backtest. Traffy silence is **unreported**, not observed dry.

The useful present role for this physical score is a **separate scenario overlay** to inspect alongside report risk. To train a joint predictor with a defensible physical contribution, the missing target is event-matched **urban wet extent or road-water depth**, including observed non-water inside a verified sensor footprint, plus drainage/outfall state. Scaling the same crop-dominated satellite target alone does not resolve the report-site mismatch.

Reproduce with [the transfer script](flood_lat_krabang_traffy_transfer.py): `.venv/bin/python flood_lat_krabang_traffy_transfer.py`. The [result JSON](artifacts/flood_bangkapi_surface/lat_krabang_edge_flow/traffy_2025_transfer.json) records all scenario/control/blend arms, period counts, cover mix, and selected choice. The script checks that its faster local CNN crop reproduces the frozen full-grid CNN output.
