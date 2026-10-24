# Synthetic runoff-route pretraining for the isolated-site CNN

**Run 23 October 2026.** This tests whether a CNN trained first on a spatial
terrain-routing task transfers better to Bangkok report sites with no own or
eight-adjacent-cell report in the prior year. The synthetic task is a training
exercise, **not observed water extent or a validated runoff simulation**.

## Result

All four methods use the same five held-out district folds, Q4 2025 dates,
250 m cells, and daily inspection budget. The budget selects 10% of cells in
each held-out fold per day, reserving 20% of those slots for strictly isolated
sites. The two CNNs have the same LiDAR, cover, canal, road, and pre-day report
inputs and the same report-training examples; only route pretraining differs.

| Ranker | Isolated positive cells found / 332 | Touching positive groups found / 158 | All report-positive cell capture |
| --- | ---: | ---: | ---: |
| Distant report history, fixed formula | **65 (19.58%)** | **44 (27.85%)** | 42.26% |
| CNN trained from scratch, with LiDAR | 48 (14.46%) | 30 (18.99%) | 42.02% |
| **LiDAR route-pretrained CNN** | **50 (15.06%)** | **31 (19.62%)** | **42.05%** |

Pretraining gains only **2 cells and 1 touching group over the matched scratch
CNN**, and trails the simple distant-history score by **15 cells and 13
groups**. Fold-level isolated-cell hits for distant history / scratch /
pretrained were **21/7/5, 10/17/21, 17/5/4, 7/12/13, 10/7/7**. The
pretrained CNN improved two folds and worsened two.

The same pretrained CNN looked stronger in **January–September 2025 source
districts**: it found 534.78 of 3,508 isolated positive cell-days (15.25%),
versus 474.18 (13.52%) for scratch and 463.18 (13.20%) for distant history.
The reversal on excluded Q4 districts is the important transfer result. This
comparison changes both geography and season, so it does not identify which
shift caused the failure.

## What was trained

1. Average the cached BMA LiDAR DTM onto the established 250 m Bangkok grid.
   Build a steepest-lower-neighbor D8 link for each valid cell, with a 0.05 m
   minimum and 5 m maximum drop. **25,804 of 25,812** candidate cells have
   valid terrain. The cached export has about 13 m median source pixels and
   is averaged to 250 m; it is not a vintage-matched
   archive for the 2023–2025 labels.
2. Generate spatially varying synthetic rain pulses. Multiply them by a crude
   runoff-source factor from ESA WorldCover built fraction. Route each source
   through up to five downhill cells with a 0.72 per-hop decay. The target is
   `log1p` of **arriving upstream source**, excluding the cell's own source.
   These choices make a known synthetic learning task; they are not measured
   hydrologic parameters.
3. Train the existing 15 × 15-patch `SpatialNovelCNN` for 100 synthetic
   batches per fold using only source-district target cells. Its synthetic
   loss fell in every fold. On three new synthetic scenarios in excluded
   districts, prediction-versus-D8-target Spearman ranged **0.305–0.539**.
   This shows some transfer on the synthetic target; without an equal-reach
   distance control it does **not** establish that the network learned
   topological flow rather than general spatial proximity.
4. Transfer the static-feature entry weights and spatial convolution blocks
   into a fresh report ranker. Zero the synthetic-source input weights and
   reset the prediction head because the synthetic source is not a complaint
   history channel. Fine-tune for two epochs on 2023–2024 source-district
   report examples, with exactly the same sampling and loss as the matched
   scratch CNN. Eligible target and adjacent report-history channels are zero
   in every training and scored patch.

Canal and road geometry are **inputs**, but the synthetic target uses only
terrain paths and built-cover source weights. It contains no canal hydraulic
direction, pipe connection, pump, gate, or tide state. The report target is a
**positive complaint site**, while an unreported comparison cell is unknown,
not verified dry. The model receives no measured rain at report scoring time;
this test says nothing about doubled-rain flood scenarios or calibrated water
probabilities.

## Decision

Do not replace the distant-history queue with this CNN. The source-district
validation gain did not survive the combined geographic and seasonal holdout.
More optimization against Q4 2025 would also be exploratory because that
quarter has been inspected repeatedly. The most useful next evidence would
be time-matched urban wet/dry observations and validated drainage connections;
those would let a spatial pretraining task and its downstream target describe
the same process.

## Reproduce

- Code: [route pretraining and matched fine-tuning](flood_citywide_route_pretrain.py).
- Metrics, normalization, fold results, and training curves:
  [JSON artifact](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_route_pretrain_isolated_fivefold.json).
- CNN weights: [checkpoint](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_route_pretrain_isolated_fivefold.pt).
- Command: `.venv/bin/python flood_citywide_route_pretrain.py`.

The route-direction self-check and the artifact denominator checks pass. The
frozen distant-history row remains **65/332 isolated cells and 44/158 groups**.
