# Group-level ranking with an explicit target-history mask

**Tested 23 October 2026.** This follows the earlier historical-repeat masking CNN that reached **13.529%** first-in-a-year Q4 report-positive cell capture. That number compared with **9.887%** for the earlier CNN and **11.946%** for simple nearby report history, all on the same five-fold daily queue. The 13.529% arm was examined on Q4 before selection; source-only arm selection reached **12.178%**. The question here is whether training on **connected same-day positive groups**, with a target-cell history mask, improves that result or the stricter isolated-site queue.

## Matched first-in-a-year queue

The daily budget selects 10% of candidate 250 m cells **within each held-out district fold**. Twenty percent of those inspection slots go to cells with **no own-cell report in the previous 365 days**. Across Q4 2025, this gives **1,664 first-in-a-year positive cell-days** among **7,209** positive cell-days overall. All rows below share that queue and denominator.

| Ranker | First-in-a-year cell capture | Overall cell capture |
| --- | ---: | ---: |
| Simple nearby report history | 11.946% | 44.114% |
| Earlier historical-repeat masked CNN | **13.529%** | **44.479%** |
| **Group-level loss + same center mask + repeat-site/block augmentation**, seeds 0 / 10000 / 20000 | **11.602% / 11.659% / 10.697%** | 44.035% / 44.048% / 43.826% |

The group-trained model did **not** improve the matched cell-ranking task in any of three seeds. The no-reserve fixed history queue still captures **47.113%** overall on these fold-based queues; it allocates no slots to first-in-a-year cells.

The result is also weaker on the proxy **site-group** outcome. Of **332** first-in-a-year positive cells with no adjacent prior-year report, the earlier masked CNN selected **0** and the group-trained CNN selected **1, 1 and 3** across seeds. Of **1,332** positives with adjacent history, the earlier CNN selected **225.12**; the group-trained CNN selected **192.06, 193 and 175**. Within **689 fold-scoped same-day touching groups** in that adjacent-history subgroup, group capture fell from **207.12** for the earlier CNN to **181, 181 and 163**. The few new isolated hits do not compensate for that loss. These touching groups are an approximation; the grid lacks original complaint IDs.

## Stricter isolated-site queue

An isolated candidate has **no own or eight-adjacent-cell report** in the preceding 365 days. On a separate queue with 20% of inspection slots reserved for these cells, the fixed 750 m/1.5 km/3 km distant-history score captures **65 of 332** isolated positive cell-days and **44 of 158** fold-scoped touching groups. These figures use a different candidate pool from the first-in-a-year table above.

Here, a multiscale residual CNN received group-level loss and a **3 × 3 target mask**. One training arm added first-in-a-year positives with adjacent history after masking their local history. Correction strength was chosen on January–September 2025 source districts, then scored in excluded Q4 districts. The three selected models captured **59.6, 64 and 54.8** isolated positive cell-days and **45, 46 and 38.8** groups. Cell capture trailed the simple rule in **all three seeds**. Group capture exceeded it by only one or two groups in two seeds and fell below it in the third.

In one seed, isolated-group-only training selected **59.06 cells / 45.06 groups**, and adding masked historical repeat groups plus random blocks selected **57 cells / 42 groups**; neither beat the simple score's 65 cells / 44 groups. These two arms were not repeated across seeds because the matched first-in-a-year experiment and the more promising masked-adjacent arm already showed the central transfer issue.

## What “target mask” means here

| Test | Training mask | Why inference matches |
| --- | --- | --- |
| Matched first-in-a-year queue | Zero the **center target cell** in all four raw pre-day report-history channels for every positive group member and every unreported comparison; repeat groups therefore lose real own-cell history. Also apply the earlier 10% random history-pixel dropout and 50% random 3 × 3 history blocks. Static terrain/cover/road/canal channels stay visible. | Every scored first-in-a-year cell already has zero own-cell history. Masked patch and full-grid scores were checked equal. This CNN has no smoothed-history input channels. |
| Strict isolated queue | Zero the target-centered **3 × 3** raw report history **and subtract those nine cells' exact contributions from all five Gaussian-smoothed history layers**, symmetrically for positives and comparisons. The historical-repeat arm also uses random 3 × 3 blocks with the same Gaussian correction. | Every scored isolated cell already has zero 3 × 3 prior history. An explicit Gaussian recomputation check and patch/full-grid equality check passed. |

For both tasks, the positive bag score is a temperature-scaled `logmeanexp` over all same-day touching positive members. The pairwise loss compares that **one group score** with random and high-history unreported candidates. A repeat training group has lower loss weight. Unreported comparison cells are **unknown**, not observed dry, and held-out districts' scored-day outcomes do not affect comparison sampling. Training uses 2023–2024 source districts; source validation uses January–September 2025; Q4 2025 excluded districts are the test. Q4 has been inspected repeatedly during this project, so these are exploratory results.

## Decision

**Keep the simple nearby-history score as the practical first-in-a-year reference and the distant-history score as the stricter isolated-site reference.** The historical-repeat CNN's 13.529% Q4 result remains the best exploratory matched first-in-a-year cell score among these rows, but its source-selected result was only 12.178%. Group-level loss with correctly applied target masks did not improve either matched transfer task. It also does not show that the CNN learned water flow: Traffy supplies report-positive sites, not verified wet/dry extent or drainage connections.

The next data improvement is original report IDs or timed verified water areas so a training group represents a real event, rather than merely touching buffered 250 m cells. A new season or city is needed to confirm any future model choice without repeatedly selecting on Q4 2025.

## Reproduce

- [Matched first-in-a-year training](flood_citywide_group_loss_history_mask.py) and [isolated/adjacent subgroup audit](flood_citywide_group_loss_history_mask_strata.py). The [seed 0](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_group_loss_history_mask_fivefold.json), [seed 10000](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_group_loss_history_mask_fivefold_seed_10000.json), and [seed 20000](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_group_loss_history_mask_fivefold_seed_20000.json) artifacts contain per-fold and daily metrics. Matching `.pt` files contain model weights.
- [Strict isolated-site training and exact Gaussian target mask](flood_citywide_isolated_group_mask.py). The [three-arm seed 0](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_isolated_group_target_mask_fivefold.json), [masked-adjacent seed 10000](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_isolated_group_target_mask_fivefold_seed_10000_isolated_plus_masked_adjacent.json), and [masked-adjacent seed 20000](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_isolated_group_target_mask_fivefold_seed_20000_isolated_plus_masked_adjacent.json) artifacts include source-selected correction strengths and held-out group counts.
