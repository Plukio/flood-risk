# CNN spatial history transfer: Bangkok daily report queue

**Run 22 October 2026.** This experiment asks whether a CNN can learn useful spatial relations among earlier report sites and transfer that rule to districts whose outcome labels were excluded from fitting. The target is **Traffy report-positive 250 m cell-days**, not observed flood extent or flood probability.

## Result

The CNN is a **small learned correction to the fixed recent + nearby score**. At full strength it performs poorly. Selecting the correction multiplier on source-district validation queues gives a small, consistent geographic-fold gain in Q4 2025:

| Q4 daily top-10% queue | Fixed recent + nearby | CNN with source-selected multiplier | Difference |
| --- | ---: | ---: | ---: |
| Five disjoint 10-district holdout folds, pooled; 7,209 positives | 47.11% | **47.96%** | **+0.85 percentage points**, about 61 additional positive cell-days |
| Original eight held-out districts; 978 positives | **60.23%** | 60.11% | −0.11 points |
| One citywide queue from the eight-district fit; 7,209 positives | 48.17% | **49.71%** | +1.54 points |

**Compare within a row only.** Each fold uses its own daily top-10% queue. The five-fold pooled result combines those five queues and is not the same candidate queue as the citywide row.

The five-fold gain has an exploratory **seven-day-block 95% interval of −0.48 to +2.16 points**; it does not yet establish a reliable benefit across new storms. The single citywide queue has an exploratory interval of **+0.33 to +2.85 points**, but its source 42 districts supply the CNN training labels, and the original eight excluded districts do not improve on their own. Q4 2025 has been inspected repeatedly in this project.

| Geographic fold | Report-positive cell-days | Fixed formula | CNN | Selected multiplier | CNN minus fixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,284 | 51.87% | 52.10% | 0.05 | +0.23 points |
| 2 | 1,570 | 54.20% | 55.03% | 0.10 | +0.83 points |
| 3 | 1,065 | 35.49% | 37.55% | 0.25 | +2.06 points |
| 4 | 1,767 | 43.60% | 44.69% | 0.05 | +1.09 points |
| 5 | 1,523 | 48.00% | 48.26% | 0.25 | +0.26 points |
| **Pooled fold queues** | **7,209** | **47.11%** | **47.96%** | Per-fold | **+0.85 points** |

The earlier [learned per-cell history ranker](FLOOD_CITYWIDE_LEARNED_HISTORY_TRANSFER.md) reached **47.10%** on these same five-fold candidate queues with a soft top-10% training loss. The CNN's spatial correction reaches **47.96%**. Its architecture and multiplier selection differ, so this is context for the CNN result, not a controlled architecture ablation.

## Model and split

The CNN reads a **15 × 15 cell patch**, or 3.75 × 3.75 km at 250 m resolution, around each candidate cell. Its channels are local report-positive day counts over the preceding **365, 90, 30 and 7 days**, each log-transformed, plus a Bangkok candidate-area mask. Three 3 × 3 convolution layers with dilation **1, 2 and 4** give the center score a receptive reach of about **1.75 km**. A zero-initialized final layer produces a spatial residual added to the frozen fixed formula:

`score = fixed recent+nearby history + multiplier × CNN spatial residual`

The CNN is shared across districts. It has **no district ID, same-day rain or physical-infrastructure channel**. This isolates whether learning from the *shape* of earlier report patterns adds value beyond the fixed 250 m Gaussian neighborhood. It does not learn a verified drainage graph.

Training uses **2023–2024** source-district positives. On each day it samples reported cells and unreported comparison cells, estimates the comparison-score 90th percentile, and uses a differentiable softplus penalty for reported cells below that cutoff. Unreported cells are **unknown**, not observed dry. The checkpoint is chosen by this surrogate loss on **January–September 2025** source districts. The residual multiplier is then selected from **0, 0.05, 0.10, 0.25, 0.50 and 1.00** by *actual top-10% report capture* on those source validation dates. Target-district outcome labels are excluded from both fitting and multiplier selection. Earlier reports from target districts remain available as input features, including near borders; this is transfer of the learned spatial rule, not a cold start without local history.

The patch-scoring and full-map-scoring implementations were checked to agree, including at map boundaries. The frozen raw CNN residual at multiplier **1.00** captured only **43.91%** in the pooled folds versus **47.11%** for the fixed score. Validation chose multipliers of **0.05–0.25**. The source-only calibration step is essential to the observed improvement.

## Interpretation

This is the clearest exploratory indication so far that a learned spatial relation can add a little report-site ranking value. It is **small**: about 61 additional positive cell-days across 7,209 Q4 positives in the five fold queues, with an interval including zero. The original eight-district holdout does not improve. The multiplier-calibration step was added after inspecting the raw CNN's Q4 failure, and the five folds share the same Q4 storm dates. The interval resamples weeks; it does not capture model-selection or new-district uncertainty. A new season is needed to confirm the gain.

The model cannot infer how floodwater flows, simulate doubled rain, estimate flood depth, or produce calibrated flooding probabilities. The CNN sees earlier **report patterns**. Its map could partly learn the geography of reporting rather than inundation.

Reproduce the original fit with [CNN training and scoring](flood_citywide_history_cnn_transfer.py), calibrate its strength with [source-only multiplier selection](flood_citywide_history_cnn_alpha.py), and reproduce the geographic test with [five-fold CNN transfer](flood_citywide_history_cnn_cv.py). The [original calibrated metrics](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_history_cnn_transfer_calibrated.json) and [five-fold metrics](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_history_cnn_fivefold_transfer.json) contain daily captures and intervals.
