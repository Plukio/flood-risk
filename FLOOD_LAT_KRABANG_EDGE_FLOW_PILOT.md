# Lat Krabang: testing terrain-connected wet-area scenarios

**24 October 2026.** This pilot tests the physical-continuity idea against *GISTDA-positive mapped water*, separately from the CNN that ranks Traffy complaint sites. It is a 250 m **scenario depth proxy**, not a trained or validated flood-depth forecast.

## Setup

| Part | Implementation |
| --- | --- |
| Terrain | BMA 10 m DTM covering 8.5 × 8.5 km of Lat Krabang after trimming to 34 × 34 complete 250 m cells. |
| Water storage | Each cell holds a volume; its 625 terrain samples define a storage-to-water-surface curve. Rain adds the same volume to every cell. |
| Candidate crossing | For each shared cell side, compare the 25 pairs of touching 10 m pixels. The lowest pairwise **maximum** is the candidate pass height. The number of pairs within 10 cm of that height sets an exploratory flow width. This does not establish a traversable route through either cell. |
| Flow | Water transfers from higher to lower water surface when it overtops the pass, using an uncalibrated broad-crested-flow coefficient. No water leaves the 8.5 km domain. |
| Controls | **Ordinary** cells use the maximum of adjacent mean elevations and a full 250 m side; **reversed** applies the same candidate passes but deliberately sends the computed transfer against the water-surface gradient. **Crop cover** is the direct WorldCover 2021 cropland fraction. |
| Scenario selection | Compare 20 mm/2 h, 60 mm/2 h, and 120 mm/6 h on six GISTDA dates, **2–10 October**. Pick the scenario with the highest mean source-date top-10% mapped-area capture for the candidate-pass model. **20 October** is excluded from that choice. These rainfall inputs are controlled scenarios, not observed event rain. |
| Outcome | Exact polygon–250 m cell intersection area of 20 October GISTDA mapped water. At each budget, rank all 1,156 cells and measure the share of mapped-positive area within the selected cells. Score ties use fractional expected selection. Unmapped space remains **unknown**. |

No drain, pump, infiltration, building barrier, tide, culvert, or inlet/outfall state is represented. The WorldCover layer is used for the simple baseline, not to invent infiltration rates. The saved [masked CNN report-site queue](FLOOD_CITYWIDE_CONNECTED_SCORE_SMOOTHER.md) remains a **different output and target**; its Q4 2025 Traffy percentages cannot be compared with this GISTDA-area percentage.

## Result

The candidate-pass model selected **120 mm over six hours** from the six earlier dates. Its source-date mean top-10% capture was **26.17%**, versus **78.88%** for crop cover on those dates. The 20 October test has **1.371 km²** of mapped water inside the trimmed grid, touching 84 cells.

| Ranking on 20 October | Top 5% cells | Top 10% cells | Top 20% cells |
| --- | ---: | ---: | ---: |
| Crop-cover fraction | **65.63%** | **79.37%** | **90.22%** |
| Ordinary mean-elevation flow | 14.82% | 23.93% | 33.17% |
| **10 m candidate-pass flow** | 23.56% | **32.86%** | **42.19%** |
| Reversed-flow control | 18.88% | 25.40% | 27.56% |

The finer crossing rule beats both flow controls on this **one selected scenario and date**, but remains far below the crop baseline. This is not evidence that the simulated depth is right. At 10 m raster resolution, **88.17%** of the mapped pixels fall in WorldCover class 40 (cropland), while cropland covers **18.78%** of the grid. The GISTDA outcome here predominantly tests mapped agricultural water, not urban street flooding.

The ranking depends strongly on the controlled rain input. On 20 October, the candidate-pass top-10% capture is **7.23%** at 20 mm/2 h, **15.97%** at 60 mm/2 h, and **32.86%** at 120 mm/6 h. At 60 mm/2 h, the ordinary control is **35.22%**, above the candidate-pass result. Thus the physical advantage is **not robust across the tested scenarios**. The 2–10 October products are nearby in time and share spatial patterns with 20 October; this is only an exploratory event holdout, not an independent cross-city or cross-season validation.

## Numerical and data checks

- A simple synthetic slope test sends water downhill under the candidate-pass rule and uphill under the reversed control; both conserve added rain volume.
- The real-domain run conserves its rainfall volume to reported floating-point precision. A 120-second step generated a non-converged pattern and up to **2.24 m** of mean cell depth under 120 mm rain. Reducing the step to 30 seconds gives a maximum of **0.810 m**; compared with a 10-second run, mean depth differs by only **0.096 mm** across cells and maximum depth by **0.185 mm**. The published map and table use 30 seconds.
- GISTDA polygons are positive detections only. Their observation footprint and acquisition time have not been independently verified. The 2026 rain series is therefore not used to claim event-matched hindcasting.
- The lowest boundary pair can be an isolated notch. An unseen wall, kerb, culvert, or route through the cell may change or block that connection. Assuming one shared water surface throughout each 250 m cell is also coarse.

## Reading and reproducing the artifacts

- [Map](artifacts/flood_bangkapi_surface/lat_krabang_edge_flow/holdout_20261020_map.png): mapped water, crop fraction, ordinary flow, candidate-pass flow. The apparent blank area in the GISTDA panel is **unknown**, not confirmed dry.
- [Result JSON](artifacts/flood_bangkapi_surface/lat_krabang_edge_flow/result.json): date selection, all tested scenarios, capture budgets, category mix, water ledger, and numerical checks.
- [GeoTIFF](artifacts/flood_bangkapi_surface/lat_krabang_edge_flow/holdout_selected_saddle_depth_proxy_250m.tif): modeled mean depth for the selected controlled scenario, in metres, EPSG:32647. It is **not observed depth**.
- [Script](flood_lat_krabang_edge_flow_pilot.py): reproduce with `.venv/bin/python flood_lat_krabang_edge_flow_pilot.py` using the cached open DTM, WorldCover, and GISTDA files.

The next useful physical test is a mapped **urban** wet-water event with known observation coverage, then an explicitly connected 10 m or hydraulic model with drainage/outfall state. Adding this candidate-pass score to the complaint CNN now would mix two targets without evidence that it helps report-site ranking.

That direct check has now been run: the [Lat Krabang Traffy transfer](FLOOD_LAT_KRABANG_TRAFFY_TRANSFER.md) finds **no Q4 2025 report-site gain** from the tested physical score or source-selected CNN blend.
