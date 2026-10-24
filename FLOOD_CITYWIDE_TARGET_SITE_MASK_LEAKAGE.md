# Does synthetic center masking leave a repeat-site shortcut next door?

**Tested 23 October 2026.** The label is a 100 m Traffy complaint buffer touching a 250 m cell on a day. A single complaint can therefore mark neighboring cells. This test asks whether a CNN trained to treat a repeat site as a synthetic first-in-a-year site can still identify it from reports in the eight immediately adjacent cells.

## Result

**Yes, the synthetic examples have a strong adjacent-history shortcut, and removing it changes the learned ranking.** In 2023–2024, **17,320 of 17,331** repeat-positive cell-days (99.94%) had a report in an adjacent cell in the preceding 365 days. The corresponding figure for genuine first-in-a-year positives was **5,092 of 6,823** (74.63%). In Q4 2025, the shares were **5,544 of 5,545** (99.98%) and **1,332 of 1,664** (80.05%). An adjacent report does **not** prove the same original complaint generated both cells; the gridded training data retain no report identifier.

On five held-out district folds in Q4 2025, a CNN trained with a **3 × 3 history mask centered on each synthetic target** captured **10.18%** of first-in-a-year report-positive cell-days. A matched CNN with the same-sized mask placed randomly in the patch captured **13.04%**. The centered mask was lower in all five folds. Its difference from the matched control was **−2.86 percentage points**, with a paired seven-day-block 95% interval of **−4.62 to −1.31 points**. The interval resamples dates only; it does not cover new districts, training seeds, or repeated selection on this Q4 test set.

| Novel-slot ranker at the same inspection budget | All Q4 report-positive cell-days captured | First-in-a-year positives captured |
| --- | ---: | ---: |
| Fixed recent + nearby history, no novel reserve | 47.113% | 0% |
| Simple nearby history, 20% novel reserve | 44.114% | 11.946% |
| Matched **random-location 3 × 3 mask** CNN | 44.368% | 13.044% |
| **Target-centered 3 × 3 mask** CNN | 43.707% | 10.184% |
| Mask location selected on source districts | 44.166% | 12.171% |

The total daily queue contains the top 10% of candidate cells **within each held-out fold**. For the last four rows, 20% of its slots are reserved for cells with no own-cell report in the prior 365 days. This is 2% of all candidate cells. The denominators are **7,209** report-positive cell-days overall and **1,664** first-in-a-year positives. The old synthetic-center-plus-random-block model scored **44.479% overall / 13.529% novel** on this same Q4 task; that best-arm figure was selected after inspecting Q4 and is not a frozen benchmark for choosing the new mask location.

## Where the difference comes from

| First-in-a-year positive cells | Denominator | Simple nearby history | Random-location mask CNN | Target-centered mask CNN |
| --- | ---: | ---: | ---: | ---: |
| **No** report in any of eight adjacent cells in prior year | 332 | 0 / 332 | 0 / 332 | **10 / 332 (3.01%)** |
| Report in at least one adjacent cell in prior year | 1,332 | 198.78 / 1,332 (14.92%) | **217.06 / 1,332 (16.30%)** | 159.46 / 1,332 (11.97%) |

The centered mask finds **10 isolated positives** that the other two rankings miss, but loses about **58 positives** where adjacent history is present. The latter group dominates the Q4 first-in-a-year target. This supports the concern that the earlier synthetic-mask improvement was substantially tied to immediate neighboring report history. It does not show that all such use is spurious: prior nearby complaints are genuinely available before prediction and can indicate a related problem site.

Source-district January–September 2025 validation selected the random-location arm in **four** folds and the centered arm in **one**. That preselected combination scored **12.171% novel** versus **11.946%** for the simple nearby score; its paired interval for the difference is **−2.45 to +3.15 points**, so a reliable novel-site gain is not established. The no-reserve fixed history queue still has the highest *total* report capture because reserving slots for zero-own-history cells trades away repeat-site picks.

## Matched design and limits

- Both arms use the same 15 × 15 CNN patches, four strictly pre-day report-history channels, source-normalized terrain/road/canal/cover channels, 2023–2024 training pairs, initialization, seed, optimizer, two epochs, 10% random history-pixel dropout, and five geographic folds.
- Synthetic positives are repeat sites with own-cell history zeroed. Every synthetic positive/comparison pair gets the same 3 × 3 history-only hole; half of actual first-in-a-year training pairs also get one. The arm changes **only the hole location**: target-centered versus random. The paired comparison receives the same hole, so hole presence alone cannot signal a positive label.
- Model choice uses January–September 2025 **source districts**; evaluation uses Q4 2025 **excluded districts**. Q4 has been inspected repeatedly during this project, so these results remain exploratory.
- The model ranks **reported cell-days**, not surveyed flood extent. Unreported cells are unknown, not verified dry. No rain response or hydraulic flow is established by this test.

The practical reading is to treat the old random-block gain as mostly a **nearby-history ranking result**, with a small sign of spatial transfer from the centered arm at isolated sites. A future cold-start claim should report performance separately for cells with **no own or adjacent prior-year reports**, and should be confirmed on a new season.

A later [joint satellite + urban CNN holdout](FLOOD_JOINT_TARGET_SITE_MASK_HOLDOUT.md) repeats the centered/random comparison with satellite auxiliary targets, urban-only controls, independent BMA positive sites, and held-out satellite scenes. A large isolated-site gain in its first seed shrinks or disappears in two additional seeds, while satellite heads show no corresponding mask gain.

## Reproduce

```bash
.venv/bin/python flood_citywide_target_site_mask_test.py
.venv/bin/python flood_citywide_target_site_mask_stratify.py
```

The first script saves the five-fold queues and CNN weights to `artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_target_site_mask_fivefold.json` and `.pt`; it checkpoints completed folds. The second saves the adjacent-history audit and subgroup counts to `artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_target_site_mask_adjacent_strata.json`.
