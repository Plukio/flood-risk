# Target-centered masking in the joint satellite and urban CNN

**Tested 23 October 2026.** This is a matched experiment on the existing `MultiTargetCNN`: does hiding immediate prior-report history around a training target help the shared model find first-in-a-year urban report sites, and does it help held-out satellite water rankings? It compares two joint CNNs and two urban-only controls on the same architecture and urban training draws, then repeats the comparison with **three seeds**.

**Result:** The large isolated-site gain in the first seed does **not** repeat. Across three seeds, target-centered masking captures **15, 4, and 6** of the same 220 isolated Q4 report-positive cells outside the urban training patches, versus **1, 3, and 6** for matched random masking. It captures fewer of **all** first-in-a-year positives in every seed. On held-out GISTDA and EOS satellite maps, the centered arm also ranks mapped water below the random arm in every seed. There is no stable predictive win from this mask recipe.

## What was held out

| Outcome | Holdout | Why it is informative | Main limit |
| --- | --- | --- | --- |
| Bangkok Traffy flood complaints | Q4 2025; additionally report the 13,302 candidate cells outside the Bang Kapi and Nong Chok urban training patches plus 750 m | Temporal and geographic transfer of the urban head | Reports are positive problem-site proxies, not verified flood extent; Q4 has been inspected previously |
| BMA road-flood records | 89 conservatively located Q4 2025 positive site-days across 30 districts | Independent urban reporting source with recorded depth | 2025 event positions are approximated by matching road names to a 2026 BMA inventory |
| GISTDA mapped water | Eight 2026 Pathum Thani patch-dates | Province/region withheld from all satellite fitting | Mostly broad mapped water, often crop-area; eight dates are correlated storm observations |
| EOS-RS Sentinel-1 proxy | Two Bangkok 2026 patches | EOS water labels withheld from fitting | The same Bangkok patch geography supplies urban training; radar water is not a street wet/dry survey |
| UNOSAT mapped water | Two wet 2021 Pathum Thani dates | Separate satellite provider in a withheld region | Only two positive dates; other Pathum Thani UNOSAT layers have no mapped wet area |

Urban reports and BMA records used for fitting stop at **31 December 2024**. The joint arms train on 134 GISTDA patch-dates, 54 UNOSAT 2021 scenes, one UNOSAT 2024 scene, and one external EOS 2025 scene, plus matched urban draws. Both joint arms use the same 3,072 scheduled steps: **1,491 Traffy pairs, 323 BMA updates, and identical satellite scene draws**. The urban-only controls replay that schedule and the same urban samples, skipping the satellite updates. All four arms use the same normalization from satellite *training* regions.

For a Traffy update, one report-positive cell and one unreported comparison cell are scored by a relative pairwise loss. An unreported comparison is **unknown**, not observed dry. Own-cell prior report history is hidden in both inputs. Each gets one additional 3 × 3 history-only hole, either centered on its scored cell or placed randomly in its patch. The positive and comparison receive symmetric treatment, avoiding hole presence as a direct label cue. Satellite heads retain source-specific positive-area losses; the BMA head retains its location/depth loss. The architecture, task draws, seed, optimizer, and masks' size are matched within each pair of arms.

## Bangkok urban report ranking

Each day, the queue selects **10%** of candidate cells. In every row below except the fixed no-reserve history queue, **20% of inspection slots** are reserved for cells with no own-cell report in the preceding 365 days. The two right columns are subgroups of the same queue. “Isolated” means no own or eight-adjacent-cell report in that prior year; it does not mean no report farther away.

**Outside the two urban training patches and their 750 m buffer:** 3,605 Q4 report-positive cell-days, including 959 first-in-a-year and 220 isolated positives. The full queue table below shows **seed 18319**; the next table shows sensitivity across all three matched seeds.

| Queue | All report positives | First-in-a-year positives | Isolated positives |
| --- | ---: | ---: | ---: |
| Fixed recent + nearby history, no reserve | **46.08%** | 0% | 0% |
| Simple nearby history, 20% reserve | 43.64% | 10.32% | 0 / 220 |
| Joint CNN, random 3 × 3 mask | **44.05%** | **11.89%** | 1 / 220 (0.45%) |
| Joint CNN, target-centered 3 × 3 mask | 43.81% | 10.95% | **15 / 220 (6.82%)** |
| Urban-only CNN, random mask | 43.42% | 9.50% | 1 / 220 (0.45%) |
| Urban-only CNN, target-centered mask | 43.50% | 9.81% | 9 / 220 (4.09%) |

| Seed | Joint random: isolated / 220 | Joint centered: isolated / 220 | Joint random: all first-in-a-year / 959 | Joint centered: all first-in-a-year / 959 |
| ---: | ---: | ---: | ---: | ---: |
| 18319 | 1 | **15** | **114** | 105 |
| 18320 | 3 | **4** | **82** | 80 |
| 18321 | 6 | 6 | **113** | 104 |

In seed 18319, the 15 isolated hits are 15 distinct cells in **10 districts**, on **eight dates** and in **10 same-day adjacent-cell groups**. That seed's seven-day-block interval for centered minus random is **+3.55 to +9.53 percentage points** on isolated sites. It resamples dates **within that one trained seed** and therefore cannot describe seed sensitivity; the other two seeds give only **+1 and 0 cells**. The centered arm's loss across the full first-in-a-year group is more consistent: **−9, −2, and −9 cells** versus random. The no-reserve history queue also remains the strongest total-capture benchmark.

Satellite auxiliary training does **not** establish the isolated-site improvement. With the same centered mask and urban draws, the joint arm finds **6, 1, and 0 more isolated cells** than the urban-only control across the three seeds. The joint *random-mask* arm finds **22.92 more, 10 fewer, and 8 more** first-in-a-year positives than its urban-only control. These mixed outcomes do not support a stable satellite-training gain for the urban ranking. The joint model also uses 2026 satellite labels when retrospectively scored on Q4 2025.

The all-Bangkok citywide queue has 7,209 report-positive cell-days, including 1,664 first-in-a-year and 332 isolated positives. In seed 18319, joint centered captures **20 / 332 isolated** versus **7 / 332** for joint random; the centered arm trails on all first-in-a-year positives (**175 versus 194**). The citywide queue and the outside-patch queue use different candidate sets and must be compared within their own rows.

## Other-source holdouts

Satellite values are **mapped positive-area capture** in the top 10% of each 80 × 80 patch, pooled by positive area across dates. Transparent pixels are unknown, not dry. Crop cover is a deliberately simple static comparator. The table shows **seed 18319**.

| Held-out satellite target | Joint random mask | Joint centered mask | Crop-cover rank |
| --- | ---: | ---: | ---: |
| Pathum Thani GISTDA, eight dates | **56.37%** | 55.51% | 55.83% |
| Bangkok EOS-RS, two patches | 9.28% | 5.40% | **14.94%** |
| Pathum Thani UNOSAT, two wet dates | **69.76%** | 68.69% | 67.08% |

Across seeds **18319, 18320, and 18321**, centered minus random mapped-water capture is **−0.86, −0.22, and −0.87 percentage points** for Pathum Thani GISTDA, and **−3.88, −1.41, and −2.50 points** for Bangkok EOS. UNOSAT's two wet dates give **−1.07, −9.28, and +0.27 points**. There is no consistent satellite benefit. Importantly, the present satellite heads **do not receive prior-report history**; the mask changes them only through shared encoder updates during urban training. This test measures indirect transfer, not satellite-map completion or the appearance of new water within a satellite sequence. Pathum Thani's eight GISTDA dates are repeated views of correlated flood events, not eight independent provinces. On the Bangkok EOS proxy, both arms trail crop cover in all three seeds.

On the **89 independent BMA road-flood site-days**, seed 18319's joint centered queue captures **28**, joint random **26**, simple nearby history **24**, and the no-reserve fixed queue **29**. Only **five** BMA site-days have no own or adjacent Traffy history. Across the three seeds, centered captures **1, 0, 0** of those five and random captures **0, 1, 0**. These five approximate site locations are far too few to confirm cold-start transfer. Reported depth-weighted results and each source's subgroup counts are in the evaluation artifacts.

Before the joint run, the earlier five-fold `SpatialNovelCNN` was also scored on these 89 BMA site-days using **separate held-out-fold queues**. Its target-centered arm captured **31.6 / 89** versus **28 / 89** for matched random masking, but **neither captured any of the five isolated BMA site-days**. Those per-fold queue percentages are not directly comparable with the joint model's citywide queue. [Five-fold BMA transfer artifact](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_target_site_mask_bma_transfer.json).

## Reading and next test

Target-centered masking **sometimes** changes which isolated Bangkok complaint sites the joint CNN finds, but the size of that gain varies sharply by seed and does not extend to the full first-in-a-year group. The fixed recent + nearby history queue remains the practical benchmark for total report capture. These are **report-ranking results**, not proof that the network learned how water physically emerges. Neither the satellite map holdouts nor the BMA isolated-site check confirms a transferable centered-mask gain. A separate isolated-site queue would need its routing, slot budget, and mask recipe selected on source validation, then a new-season test.

To test the masking idea **directly on satellite emergence**, a later model would need a dated *prior mapped-positive-water* channel plus an explicit observation/availability channel for each source. It should mask the target's 3 × 3 prior-positive neighborhood symmetrically in positive and comparison inputs, then evaluate **first mapped positives** in an entirely held-out region and storm. A prior transparent pixel must remain unknown; it cannot be called observed dry or proven new flooding.

## Reproduce and inspect

```bash
.venv/bin/python flood_joint_target_site_mask_train.py
.venv/bin/python flood_joint_target_site_mask_eval.py
```

Use `--force` on the training command to retrain all four arms instead of reusing their saved checkpoints. The [four checkpoint training records](artifacts/flood_open_weak_train/multi_target/target_site_mask/training_summary.json) and [full holdout evaluation](artifacts/flood_open_weak_train/multi_target/target_site_mask/holdout_evaluation.json) contain task counts, daily urban capture, paired intervals, isolated hit cells, BMA site groups, and each satellite scene's score.

Repeat the joint arms with `--seed 18320` and `--seed 18321` and a separate `--output-dir` for each, then evaluate each with `--checkpoint-dir` pointing to that directory. The [three-seed summary](artifacts/flood_open_weak_train/multi_target/target_site_mask/three_seed_summary.json) and [aggregation script](flood_joint_target_site_mask_seed_summary.py) preserve the matched comparisons. These runs reuse the **same** outcome dates and places, so three seeds test model instability, not three independent validation sets.
