# Terrain-conditioned continuity loss probe

**Tested 24 October 2026.** This addresses the proposed case where three neighboring cells have high CNN scores while a lower center cell has a low score. It is an *auxiliary training loss* on the existing masked CNN's **report-site ranking logits**, not a law of flooded depth or a calibrated wet/dry probability.

For a center cell `c`, the loss activates when at least three of its four side neighbors have valid BMA DTM means between `z_c` and `z_c + 0.25 m`. Let `s_(3)` be the third-highest score of those eligible neighbors, and `s_c` the center score. The added term is `mean(max(0, s_(3) - s_c - 0.1)^2)` with weight 1, alongside the original positive-versus-unreported pairwise ranking loss. The 0.1 margin is in arbitrary logit units. We also tested an otherwise identical loss using all valid side neighbors, and a matched continuation with no added loss. All arms start from the same saved masked-CNN checkpoint and use matched samples and masks.

The full grid has **25,812** candidates; **4,673** have three terrain-eligible side neighbors under this coarse rule. This counts *possible* incoming sides, not proven open passages. The model still has no measured water-surface height, hard-barrier map, or verified wet/dry labels. Its report ranking score can be low where flood occurs but nobody reports it.

## Limited five-fold training probe

Each district fold continued training for one epoch on **90 sampled positive training days** from 2023–2024; January–September 2025 source districts chose among the original checkpoint and the three continuations. Q4 2025 districts remained excluded from fitting and selection. The daily queue reserves 20% of a fixed top-10% area budget for first-in-a-year cells.

| Model | Q4 first-in-a-year capture, 1,664 positives | Overall capture, 7,209 positives |
| --- | ---: | ---: |
| Original masked CNN | 13.529% | 44.479% |
| Matched continuation, no added loss | 13.362% | 44.441% |
| Ordinary three-neighbor continuity loss | 13.542% | 44.482% |
| Terrain-conditioned continuity loss | 13.510% | 44.475% |
| Source-selected arm per fold | 13.422% | 44.455% |

Source validation chose **no terrain-loss arm in any fold**: it chose the no-loss continuation four times and the original once. Terrain minus no-loss has a seven-day-block Q4 interval of **−0.11 to +0.40 percentage points**; terrain minus ordinary is **−0.22 to +0.06 points**. These exploratory intervals omit geographic and model-selection uncertainty, and Q4 was examined before this probe. The 90-day subset is a deliberately limited continuation, so this does not rule out other loss weights or full retraining. It gives no reason to promote this penalty now.

The [loss prototype](flood_citywide_continuity_loss.py) has a gradient unit check, matched training arms, fold selection and the [diagnostic artifact](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_mask_cnn_continuity_loss_diagnostic.json). A physically grounded version would use an estimated **water-surface height and edge saddle** to determine when water could cross; it would need verified observations of both wet and dry locations to test the resulting extent.
