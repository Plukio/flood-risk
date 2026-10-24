# Masked map completion and the unchanged Bangkok main test

Run on 22 October 2026. This tests whether hiding part of a GISTDA mapped-water target creates a useful spatial learning task **without changing the original Bangkok test**. The main test still ranks mapped positive area across 250 m cells in the fixed, entirely held-out Nong Chok patch, using terrain, canals/roads, land cover, and rain. At main-test inference, **the same-day GISTDA map is entirely hidden**. The additional completion test reveals mapped water outside a hidden 5 × 5 km block and ranks cells within that block.

Unmapped cells are unknown, not verified dry. Neither score is a flood probability.

## Training and selection

- The first joint U-Net mixed fully hidden maps and masked 20 × 20-cell blocks from all 20 external patches. It was selected on later dates from those same patches. This was an initial check, with **153 positive training patch-days** after adding six 2025 dates.
- The follow-up reserved **Pathum Thani, Chachoengsao, Ayutthaya, and Nakhon Pathom** as four wholly unseen external validation patches. The remaining **16 patches supplied 125 positive training patch-days**; validation had 30 positive patch-days. The target Nong Chok patch was absent from training, feature normalization, and epoch selection.
- The follow-up first trained a crop-prior U-Net to complete randomly placed positive-water blocks. It then fine-tuned for the fully hidden main task, with masked completion as a smaller auxiliary loss. A second U-Net trained from scratch on the main task provides a control. The main output starts at the exact crop-cover ranking and learns a bounded spatial correction. Epoch selection used top-10% capture in the four withheld external patches, never Bangkok.

The block masks generate many *views* of each image, but not independent storms or independent ground-truth maps.

## Fixed Bangkok results

The measure is the mapped-water area caught in the highest-ranked 10% of candidate cells. In the main test, the budget is 10% of Bangkok cells inside Nong Chok. In the completion test, it is 10% **within each hidden block**; these two columns have different candidate sets and should only be compared within their own column.

| Ranking | Main: full map hidden | Additional: hidden block, rest visible |
| --- | ---: | ---: |
| Simple WorldCover crop fraction | **44.32%** | **38.34%** |
| First joint masked/full U-Net | 36.97% | 32.90% |
| Geographic validation: masked pretraining only | 40.85% | 32.54% |
| Geographic validation: pretrain then main-task fine-tune | 38.88% | 30.35% |
| Geographic validation: main-task CNN from scratch | 32.74% | — |

The follow-up completed **121 positive hidden blocks** across eight Bangkok dates. Revealing the same-day map outside each block improved the masked-pretraining model from **32.20% to 32.54%** within hidden blocks. The gain is only 0.35 percentage points, and crop alone remains 5.80 points higher. The model has not learned useful map completion on this holdout.

On the four external validation patches, main-task fine-tuning improved top-10% capture from **29.30% for crop** to **32.43%**. Yet the same chosen model fell to **38.88% in Bangkok**, versus **44.32% for crop**. This is a concrete geographic transfer failure. The scratch control reached 31.85% on external validation and 32.74% in Bangkok. Masked pretraining helped relative to scratch, but did not beat the simple Bangkok baseline.

Replacing the chosen model's rain channels with their training means changed Bangkok capture from **38.88% to 39.19%**. Doubling the rain-intensity channels changed some selected cells (top-10% Jaccard overlap **0.888**), but there is no observed doubled-rain event to validate that change. Full-day rain may also include hours after the image acquisition. These numbers do not establish rain-scenario skill.

## What to change before claiming a better main model

1. **Keep the whole-map-hidden test as the model-selection target.** Use geographically withheld cities or provinces for selection, and retain crop fraction as the reference. The masked score is an auxiliary diagnostic, not a substitute for main-task performance.
2. **Use completion only to pretrain spatial features.** Keep a strong crop skip connection and a small residual for the main head. The current main-task fine-tune still shifted rankings away from Bangkok positives; a learned correction should be promoted only after it improves several unseen cities, not one validation group.
3. **Increase independent event and target information.** Extra masks from one image do not teach how water changes with rainfall. The current GISTDA positives emphasize persistent agricultural water. A rain-response target needs acquisition-time-matched precipitation, comparable sensor coverage across dates, and observed water extent on built surfaces. Without these, the model can learn crop location while giving plausible but unverified responses to rain.

For this mapped-water target, **crop cover remains the winning Bangkok ranking**. A future model can retain crop as its fallback, but a fallback is not evidence that its rain-driven correction works.

## Reproduce

The [province-expansion report](FLOOD_GISTDA_PROVINCE_EXPANSION.md) describes the source caches and fixed Bangkok evaluation. Training scripts and raw metric artifacts are:

- [`flood_gistda_masked_completion.py`](flood_gistda_masked_completion.py) and [`gistda_masked_completion_bangkok_metrics.json`](artifacts/flood_gistda_multiregion/gistda_masked_completion_bangkok_metrics.json).
- [`flood_gistda_masked_transfer.py`](flood_gistda_masked_transfer.py) and [`gistda_masked_transfer_bangkok_metrics.json`](artifacts/flood_gistda_multiregion/gistda_masked_transfer_bangkok_metrics.json).

Run each script with `.venv/bin/python` after building the cached GISTDA, rain, terrain, OSM, and WorldCover data.
