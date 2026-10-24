# Pushing spatial masking: synthetic zero-history sites

**Run 22 October 2026.** This follow-up asks whether stronger masking helps a spatial CNN rank **first-in-a-year Traffy report sites**. It uses the same five held-out Bangkok district folds and the same daily inspection budget as the [spatial novel-queue test](FLOOD_CITYWIDE_SPATIAL_NOVEL_QUEUE.md). It does not predict verified flooding or water flow.

## Result at a fixed 20% reserve

The total daily queue still selects **10% of candidate 250 m cells**. In the rows below, **20% of those inspection slots**—2% of all candidate cells—are reserved for cells with no own-cell report in the preceding 365 days. The remaining slots use the frozen recent + nearby score. All novel rankers compete for the **same reserved slots**; the Q4 2025 denominator is 7,209 report-positive cell-days overall and 1,664 first-in-a-year positive cell-days across five excluded-district queues.

| Novel-slot ranker | Overall capture | First-in-a-year capture | Same-day adjacent-cell cluster capture |
| --- | ---: | ---: | ---: |
| Simple nearby-history score | 44.114% | 11.946% | 57.499% |
| Earlier CNN, 10% random neighboring-history masks | 43.639% | 9.887% | 59.313% |
| CNN plus **synthetic center masking** | 44.332% | 12.892% | 59.123% |
| CNN plus synthetic center masking **and random 3 × 3 history blocks** | **44.479%** | **13.529%** | **59.386%** |
| CNN mask recipe selected separately on each fold's source districts | 44.168% | 12.178% | 59.273% |

The synthetic-center-plus-block variant is **+0.365 points overall** and **+1.583 points for first-in-a-year positives** versus simple nearby history, or about **26 extra positive cell-days** on this pooled retrospective queue. Its seven-day-block 95% intervals for CNN minus simple are **−0.197 to +1.063 points overall** and **−0.924 to +4.262 points** on the first-in-a-year subgroup. These date-only intervals include zero and do not cover model-selection uncertainty or new districts.

**Source-only recipe selection is the more relevant result.** Each fold selected the mask arm with the highest first-in-a-year capture on **January–September 2025 source districts**: center-only in fold 1, center-plus-block in folds 2–4, and the earlier CNN in fold 5. That rule yielded only **+0.054 points overall** and **+0.232 points** on the first-in-a-year subgroup versus simple nearby history—about **four extra positive cell-days**. Its date-only intervals are **−0.460 to +0.724 points overall** and **−2.161 to +2.891 points** on the subgroup. Fold 5 illustrates selection instability: source validation chose the earlier CNN, while Q4 favored the newly masked variants. The all-fold center-plus-block row is a **post hoc Q4 comparison**, not the source-selected operational recipe.

The no-reserve fixed-history queue still captured **47.113% overall** and **0%** of first-in-a-year positive cell-days in these folds. Any reserved-slot policy trades established-site capture for first-in-a-year discovery. Source validation in the earlier quota test selected **no reserve** when maximizing overall report capture. The mask ablation holds the 20% reserve fixed to test *which novel-slot ranker is better*; it does not establish that the reserve itself is worthwhile.

## How the masks work

All arms use the same candidate-centered [spatial CNN](flood_citywide_spatial_novel_queue.py): 15 × 15 cells, pre-day 7/30/90/365-day report history, source-normalized SRTM terrain, OSM road/canal geometry and WorldCover cover. The CNN composes five local convolution steps. Scored first-in-a-year cells naturally have zero own-cell report history, while their physical center features remain visible. The training target is a **future report-positive cell-day**, not reconstruction of an artificially hidden map pixel.

| Arm | Extra training examples or masks |
| --- | --- |
| Earlier CNN | Train on true first-in-a-year source positives; independently hide 10% of historical-report pixels in training patches. |
| Synthetic center | Add source training positives that **did** have prior own-cell reports. Set their four center report-history inputs to zero, so the CNN must score them from surrounding reports and physical context. True first-in-a-year positives receive greater loss weight. Keep the 10% random history masking. |
| Synthetic center + blocks | Same examples and center masking; additionally hide a random 3 × 3 block of historical-report channels in about half of training patches. |

For each day, the pairwise loss compares actual or synthetic positive source cells with **unreported zero-own-history source cells**; the latter are unknown, not verified dry. Training uses **2023–2024** only. No report from the scored day enters an input. Synthetic examples may be easier than genuinely first-in-a-year sites: their neighboring cells can still reveal repeated complaint geography. That mismatch is a likely limit on transfer.

Target-district outcome labels are excluded from fitting and arm selection, but earlier reports in those districts remain input context. Physical layers are mostly static snapshots and may not match every event date. The center-only and center-plus-block models share initialization and sampling seed within each fold; the added blocks change the subsequent masking stream. The earlier CNN is the saved checkpoint from the original run. This experiment tests synthetic example creation and block masking in combination; it does **not** identify which physical channel, if any, explains a gain.

The cluster number is a same-day adjacency proxy over 250 m positive cells, not original complaint IDs or verified distinct flood sites. Although masked CNNs score higher than simple nearby history on that diagnostic, **all reserved-slot rows remain below the no-reserve fixed queue's 61.508% cluster capture**. Q4 2025 has been inspected repeatedly in this project, so all Q4 comparisons are exploratory. None of these masks proves that the network learned hydraulic connectivity, drainage direction, or flood probabilities.

## Decision

**Keep simple nearby history as the reference novel-slot ranker.** Synthetic center masking is a credible research direction because its source-selected version no longer trails simple nearby history, but its held-out gain is tiny and uncertain. The random block addition improves the pooled Q4 result when applied everywhere; source-only selection does not recover that full gain. Confirmation requires later untouched report dates or verified sites and a predefined reserve value. For actual water-flow learning, obtain connected drainage state and timed urban wet/dry observations.

Reproduce with [mask-augmentation code](flood_citywide_spatial_mask_augmentation.py). [Per-fold and daily metrics](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_spatial_mask_augmentation_fivefold.json) and [new model weights](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_spatial_mask_augmentation_fivefold.pt) are saved.
