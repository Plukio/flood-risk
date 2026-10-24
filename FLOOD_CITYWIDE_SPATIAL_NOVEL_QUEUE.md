# Spatial CNN for first-in-a-year report sites: fixed-budget test

**Run 22 October 2026.** This experiment tested whether a more connected CNN could rank cells with no own-cell report in the preceding 365 days, using cases from other Bangkok districts and earlier years. It also tested the cost of reserving inspection slots for those cells. The target is **Traffy report-positive 250 m cell-days**, not verified flooding or water flow.

## Result

The learned CNN **did not beat a simple nearby-report score** for zero-own-history cells. At every tested reserved-slot share, it found fewer first-in-a-year report-positive cell-days and lowered overall positive-cell capture. Source-district validation selected **zero reserved slots** in all five geographic folds when maximizing total report capture.

Each Q4 2025 row below pools the **same five held-out 10-district queues**, 7,209 report-positive cell-days, 1,664 of them in cells with no own report during the previous year. The fixed total inspection budget is 10% of candidate cells **per fold and day**. A 20% reserve means 20% *of those inspection slots*, or 2% of all candidate cells, are assigned to the zero-history queue. The two methods use identical known-site slots and differ only in how they rank zero-history cells.

| Share of inspection slots reserved for zero-history cells | Novel ranker | Overall cell-day capture | First-in-a-year capture | Same-day adjacent-cell cluster capture |
| ---: | --- | ---: | ---: | ---: |
| 0% | Fixed recent + nearby history | **47.113%** | 0.000% | **61.508%** |
| 10% | Simple nearby history | **45.646%** | **6.281%** | 59.133% |
| 10% | Spatial CNN | 45.122% | 4.008% | 60.015% |
| 20% | Simple nearby history | **44.114%** | **11.946%** | 57.499% |
| 20% | Spatial CNN | 43.639% | 9.887% | 59.313% |
| 30% | Simple nearby history | **42.122%** | **17.073%** | 55.849% |
| 30% | Spatial CNN | 41.502% | 14.387% | 58.607% |

At the 20% reserve, the CNN trails the simple score by **0.475 percentage points** in overall capture and **2.059 points** among first-in-a-year positives. Exploratory seven-day-block intervals for CNN minus simple are **−0.855 to −0.174 points** overall and **−3.915 to −0.713 points** for first-in-a-year positives. The simple score's overall capture at this reserve is **2.999 points below** the no-reserve fixed queue. These intervals resample dates only; they do not capture geographic/model-selection uncertainty or repeated Q4 inspection.

The CNN has higher **adjacent-cell cluster capture** than the simple novel score at the same reserve, yet both are below the no-reserve fixed queue. This cluster diagnostic groups touching positive 250 m cells on the same day within a held-out fold and uses the maximum fractional selection weight per group. It is sensitive to complaint-buffer geometry and is **not** an original-complaint or verified-site metric. A CNN can spread its picks across groups while still missing more positive cells. This difference does not establish operational benefit.

## What was tested

The novel CNN reads a continuous 15 × 15 cell patch of **strictly pre-day** report counts over 365, 90, 30, and 7 days, plus source-normalized SRTM elevation/terrain, OSM road and canal geometry, WorldCover built/crop/tree fractions, and the candidate-area mask. Five spatial steps with 3 × 3 kernels and dilations **1, 1, 1, 2, 2** compose information through the patch to a **seven-cell, roughly 1.75 km radius**. This is denser spatial mixing than the earlier independent sparse-dilation branches. For every scored zero-history candidate, all own-cell report-history channels are exactly zero; its physical center features remain visible. Training randomly hides **10% of surrounding historical-report pixels** as augmentation.

The CNN fits only on **2023–2024 source-district report-positive cells with zero own prior-year history**, comparing them to same-day unreported zero-history source cells. Half of the comparisons are sampled randomly and half from cells highly ranked by nearby history. Unreported cells are **unknown**, not verified dry. The simple comparator is `0.6 log1p(250 m-smoothed prior-year reports) + 0.3 log1p(750 m-smoothed) + 0.1 log1p(1,500 m-smoothed)`. Known-history cells retain the frozen recent + nearby score. Quotas of **0%, 10%, 20%, and 30% of the total inspection slots** are evaluated at the same full budget.

For each of five geographic folds, all fitted weights and static normalization use 40 source districts. The training period is **2023–2024**; **January–September 2025 source districts** select the reserved-slot share; **October–December 2025** is the excluded-district test. Earlier reports from excluded districts remain input context, including near borders; their test outcomes are excluded from fitting and quota selection. The five selected shares were **0%, 0%, 0%, 0%, 0%** for both methods. Patch and full-grid CNN scores, including map boundaries, passed equality checks. Queue allocation reproduces the frozen fractional top-10% selector and the **47.113%** fixed five-fold benchmark.

## Decision and limits

**Keep the fixed history queue for a total-capture objective.** A separate zero-history reserve creates a measurable discovery trade-off, but this CNN did not make that trade-off better than a simple neighbor score. Its 10% random history masking and denser convolutions did not demonstrate a transferable spatial advantage on the current report labels. This is an architecture-plus-training comparison; it does not isolate the effect of random masking, each physical channel, or the loss function. A [follow-up synthetic center-mask test](FLOOD_CITYWIDE_SPATIAL_MASK_AUGMENTATION.md) raised exploratory Q4 novel-slot capture, but its source-selected gain over simple nearby history was small and uncertain.

"Zero own-cell history" is a grid definition, not proof of a new physical flood site; a past complaint can lie in a neighboring cell. SRTM, OSM and WorldCover snapshots may differ from the historical event dates. Mapped canals do not provide verified drainage direction, gate state, or pump operation. Traffy reports measure reporting as well as urban problems. These labels cannot establish that the CNN learned hydraulic connectivity, actual flood extent, or a doubled-rain scenario.

The next useful data improvement is **stable site-level outcomes or timed wet/dry observations and connected drainage state**. If the objective is first-in-a-year discovery despite lower total capture, the simple neighbor queue is the current reference; choose its reserve based on the value of new sites and verify findings on later, untouched storms before claiming field value.

Reproduce with [training and evaluation code](flood_citywide_spatial_novel_queue.py). [Per-fold and daily results](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_spatial_novel_queue_fivefold.json) and [saved CNN weights](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_spatial_novel_queue_fivefold.pt) are available.
