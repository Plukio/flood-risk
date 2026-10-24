# Center-masked multiscale CNN: Bangkok report-ranking transfer

**Run 22 October 2026.** This test asks whether a CNN can learn a transferable spatial correction from *neighboring* report history while the fixed score handles the candidate cell's own history. The target is a Traffy report-positive 250 m cell-day, **not** verified flooding, depth, extent, or a flood probability.

## Matched held-out result

Each row uses the **same five Q4 2025 geographic-fold queues**: ten districts are excluded from fitting and model selection in each fold, and each fold selects its own daily top 10% of candidate cells. The pooled denominator is 7,209 report-positive cell-days. A pooled fold queue is different from one citywide top-10% queue.

| Ranker | Q4 capture | Gain over fixed | Cells with no own report in prior 365 days: capture of 1,664 positives | Same-day adjacent-cell cluster capture of 1,777 clusters |
| --- | ---: | ---: | ---: | ---: |
| Fixed recent + nearby history | 47.113% | Reference | 0.00% | 61.508% |
| Earlier dilated CNN, source-selected strength | **47.962%** | +0.849 points | 0.18% | 61.148% |
| **Center-masked multiscale CNN**, source-selected strength | **47.954%** | +0.841 points | **0.00%** | **61.677%** |

The new model is **effectively tied with the earlier CNN** on the main capture measure: −0.008 percentage points, less than one positive cell-day after rounding. Its paired seven-day-block interval relative to the earlier CNN is **−0.892 to +1.007 points**. Relative to the fixed score, its interval is **+0.172 to +1.451 points**. These are exploratory date-resampling intervals. They do not account for repeated Q4 inspection, model selection across project runs, or new-city uncertainty.

| Geographic fold | Fixed | Earlier CNN | Center-masked CNN | Center-masked minus earlier CNN |
| --- | ---: | ---: | ---: | ---: |
| 1 | 51.869% | 52.103% | 52.804% | +0.701 points |
| 2 | 54.204% | 55.032% | 55.159% | +0.127 points |
| 3 | 35.493% | **37.549%** | 35.775% | −1.774 points |
| 4 | 43.599% | 44.692% | 44.765% | +0.073 points |
| 5 | 47.997% | 48.260% | 48.654% | +0.394 points |

The cluster check groups report-positive cells that touch along an edge or corner **on the same date within a held-out fold**. It reduces repeated credit when a 100 m complaint buffer touches multiple 250 m cells, but it is only a proxy: original complaint IDs and cross-day event linkage are unavailable in this grid. The masked model's cluster gain is 0.169 points over fixed and 0.529 points over the earlier CNN; this is too small to establish a useful field benefit. Cluster ties use the maximum fractional selection weight in each cluster.

**The no-own-history test is a failure for the intended transfer promise.** All three rankers use the same full queue. Among 1,664 positive cell-days with zero own-cell reports in the previous year—**23.1%** of all 7,209 positives—the new model captures none; the earlier CNN captures about three. A small residual on a history-dominated score does not promote newly reporting cells into the top 10%, even when it learns neighbor structure. "No own-cell report" does not mean a genuinely new flood site: neighboring cells may have prior reports, and a 100 m complaint buffer can touch several grid cells. Do not present this model as finding new flood sites or as solving cold-start transfer.

## Method

The frozen fixed score is `log1p(prior 365-day reports) + 0.4 log1p(prior 90-day reports) + 0.4 log1p(250 m-smoothed prior 365-day reports)`. The new spatial branch takes 15 × 15 patches of pre-day report counts for **365, 90, 30, and 7 days**, plus the candidate-area mask. Three learned 3 × 3 filters at dilation **1, 3, and 7** see neighbors at roughly 250 m, 750 m, and 1.75 km. Their center weights are fixed at zero, so the branch cannot read the candidate cell's own input; a self-check changes the center input and confirms that the score does not move. A zero-initialized head produces a residual added to the fixed score.

Training uses **2023–2024** source-district report-positive cells. On each day, it computes the model score for **every source-district candidate** and uses the actual daily top-10% boundary cell in the differentiable ranking loss. It samples up to 32 positives on that day for the gradient step. Unreported candidates set the inspection-queue cutoff; they are **unknown**, not verified dry. Checkpoint and residual-strength selection use only **January–September 2025 source districts**. The residual strength is selected from 0, 0.05, 0.10, 0.25, 0.50, and 1.00. Q4 2025 target-district outcomes are excluded from fitting and selection. Prior-day reports from target districts remain available as input features.

The source-selected strengths were **0.25, 0.50, 0.50, 0.25, 0.25** across the five folds. The old CNN scores were regenerated from its saved fold checkpoints and original source-selected strengths on the **same candidate queues**; they reproduce the original 47.962% pooled result. The fixed score reproduces 47.113%. The model also passed patch-versus-full-grid equality and center-input invariance checks.

The CNN sees report patterns only. It has no rain, drainage state, verified wet/dry extent, or physical flow network. Its ranking cannot support doubled-rain scenarios. The result does not justify replacing the simple fixed-history operational benchmark, especially for newly reporting locations.

## Other ways to improve the spatial queue

| Priority | Experiment | Why it is worth testing | Required comparison |
| --- | --- | --- | --- |
| **1** | **A separate first-in-a-year queue.** Reserve a source-validation-selected fraction of the daily inspection slots for cells with zero own-cell reports in the prior 365 days. Rank those cells by earlier reports in 250 m–1.5 km neighborhoods, then test a learned neighbor/terrain/cover model. Keep the remaining slots ranked by the fixed history score. | The main queue currently selects none of this subgroup's 1,664 Q4 positives. A previous shared pairwise history ranker selected **5.23%** of them, but lowered overall capture to **46.24%** from **47.11%**; adding static features gave **46.96%** overall with the same subgroup capture. This demonstrates a trade-off that a separate slot budget can expose explicitly. [Earlier test](FLOOD_CITYWIDE_LEARNED_HISTORY_TRANSFER.md). | At each fixed total budget, report **overall positive-cell capture, first-in-a-year capture, and cluster capture**, plus how many established-site positives were displaced. Compare a simple smoothed-neighbor score with any learned model *within the reserved slots*. Select the slot fraction using source districts before scoring target districts. |
| **2** | **Rank sites or road segments with spatial separation.** Group adjacent current-day positives for training and scoring diagnostics, and prevent a queue from spending many slots on nearby cells that describe one location. Candidate site groups or road segments must be defined before the scored day. | The 7,209 positive cell-days reduce to **1,777 same-day adjacent-cell groups** in this diagnostic. The current cell objective rewards multiple hits around one complaint buffer. | Compare unique-group capture at the **same number of inspected locations**, alongside cell capture. A same-day positive group is an evaluation proxy, not a known unique complaint or verified flooded site. |
| **3** | **Learn a spatial rule with a cold-start-specific loss.** Train a separate head only on zero-own-history candidate cells, comparing reported sites with unreported candidates on the same day. Inputs can include pre-day multiscale neighbor counts, surface cover, terrain position, roads, and canal proximity; constrain the head and test ablations. | A residual on a history-dominated score is too small to move these cells across the full-queue cutoff. A separate head can learn from analogous areas and times without competing for score scale against repeat-report sites. Static features have previously transferred inconsistently, so each added layer must earn its place. | Beat the **simple neighbor-only first-in-a-year ranker** on held-out districts and later dates. Unreported cells remain unknown, not dry. Use only features available before each scored day. |
| **Later** | **Directed physical connections**, after obtaining better drainage links and operating state. | The existing terrain-path feature reached only **12.55%** of cells at its tested reach, and OSM canal continuity does not establish water flow. [Path test](FLOOD_CITYWIDE_UPHILL_PAVED_BACKTEST.md). | Compare with equal-distance neighborhoods and reversed paths; require an added gain on locations outside the old patches. |

**Follow-up result:** the separate first-in-a-year queue and a denser CNN with terrain, cover, road and canal inputs were [tested on the same five folds](FLOOD_CITYWIDE_SPATIAL_NOVEL_QUEUE.md). At a 20% share of inspection slots, the simple neighbor score captured **11.946%** of first-in-a-year positives and **44.114%** of all positives; the CNN captured **9.887%** and **43.639%**, respectively. Source validation selected a **zero-slot reserve** in all folds for the objective of total report capture. Q4 2025 has been used for exploratory decisions; a later untouched season is needed for confirmation. Neither queue nor its labels establishes actual flood probability or doubled-rain behavior.

Reproduce with [training and matched evaluation](flood_citywide_masked_multiscale_cnn.py). [Daily metrics and fold details](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_center_masked_multiscale_cnn_fivefold.json) and [model checkpoints](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_center_masked_multiscale_cnn_fivefold.pt) are saved alongside the earlier CNN artifacts.
