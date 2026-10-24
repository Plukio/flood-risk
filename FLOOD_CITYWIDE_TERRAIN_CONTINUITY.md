# Terrain-linked continuity for the Bangkok report queue

**Tested 24 October 2026.** A nearby reported cell may be part of a larger wet area, but report absence in the next cell does not establish dry land. We tested whether coupling the saved masked CNN's **scores** across adjacent, similar-height cells improves the same first-in-a-year Traffy report queue. We also added a separate map overlay of *possible* adjacent extent around Q4 reports. Neither operation creates a verified wet label.

## Physical idea and available evidence

Water can spread from one cell to another when its water surface is high enough to pass the saddle or obstruction between them, subject to drainage and current gate/pump state. A 250 m cell mean elevation is only a coarse clue to that condition. The cached BMA DTM exports have variable source pixels of about **3.74–45.13 m** and are averaged to the project's 250 m grid. The graph has **51,009** side-adjacent candidate-cell edges; **36,182** differ by at most **0.25 m** in mean height. We have no complete map of walls, kerbs, culverts, embankments or time-matched operating state, so we cannot certify an edge as open merely because no barrier is mapped.

## Matched queue test

For each scored day and each saved fold-specific CNN, compute a one-hop average of neighboring CNN logits **only over cells that were first-in-a-year candidates before that day**. The original score and neighbor average are blended with a nonnegative weight selected separately in each fold on January–September 2025 source districts. The terrain graph weights an edge by `exp(-(absolute mean-height difference / 0.25 m)^2)`. A control gives all valid side neighbors equal weight. Both use the same cells, Q4 dates, fixed 10%-of-cells daily budget, and 20%-of-slots first-in-a-year reserve. They do not use Q4 reports as score inputs.

| Q4 ranker | First-in-a-year capture, 1,664 positives | Overall capture, 7,209 positives |
| --- | ---: | ---: |
| Nearby report history | 11.946% | 44.114% |
| Saved masked CNN | 13.529% | 44.479% |
| CNN + ordinary neighbor average, source-selected blend | **14.113%** | **44.614%** |
| CNN + terrain-weighted neighbor average, source-selected blend | **14.083%** | **44.607%** |

The terrain-weighted variant gains **0.554 percentage points** over the saved CNN on this Q4 queue, about **9.2** extra report-positive cell-days. A seven-day-block interval for that difference is **−0.55 to +1.28 points**. It trails ordinary neighbor smoothing by **0.03 points**; the interval for terrain minus ordinary is **−0.67 to +0.53 points**. These date-only intervals omit model-selection and geographic uncertainty. The previous CNN arm and Q4 dates have been examined repeatedly, so the comparisons remain exploratory. A fixed 50% terrain blend reaches **14.504%** on Q4, but it was not the source-selected choice and should not be promoted from this test.

**Interpretation:** Some local score continuity helps this report-ranking task, but this test does **not** isolate a physical terrain benefit. The CNN already sees neighboring cells, and simple smoothing performs just as well. A larger or truly hydraulic continuity model would need water-surface height or depth at the seed, higher-resolution edge saddles, mapped barriers and culverts, and event-time drainage state. If those become available, continuity should constrain possible depth/flow across an edge rather than force two cells to share a binary flood label.

## Map layer

The [interactive Q4 map](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_mask_cnn_history_interactive_map.html) now lets you switch the right map between the original CNN and **CNN + terrain neighbors**. The latter uses each fold's source-selected blend weights and reproduces the 14.083% result at 10% top area. Turn on **Show possible terrain-linked cells** to outline an unreported cell adjacent to a Q4 report cell when their mean DTM heights differ by at most 0.25 m. That outline is generated *from the observed report on that date*, is retrospective only, and is never a model input or a confirmed flood observation. The [Bang Kapi detail preview](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_mask_cnn_continuity_detail_preview.png) shows the outline and the different-picks view.

Reproduce the [graph builder](flood_citywide_terrain_continuity.py), [queue test](flood_citywide_mask_cnn_continuity.py), and [map exporter](flood_citywide_q4_queue_map.py). The [five-fold results](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_mask_cnn_terrain_continuity_fivefold.json) preserve source choices, daily captures and graph audit.
