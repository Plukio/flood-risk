# Did all-target training improve the urban ranking?

Checked 22 October 2026. The [joint CNN](artifacts/flood_open_weak_train/multi_target/multi_target_cnn_all_open_sources.pt) trained on satellite maps, Traffy reports, and matched BMA incidents/depths. I compared it with an [urban-only ablation](artifacts/flood_open_weak_train/multi_target/urban_only/urban_only_cnn.pt) that uses the **same architecture, inputs, normalization, and report/BMA labels** but receives no satellite-label gradients. Their urban training exposure was closely matched: **1,399 Traffy and 299 BMA updates** for the joint model, versus **1,411 and 317** for urban-only. The joint run has additional satellite updates.

October–December 2025 Traffy and BMA rows were excluded from both models' urban training. The metric is the share of report-positive 250 m cell-days captured by a daily **top-10% cell queue within each Bangkok patch**. Missing complaints remain unobserved, not verified dry cells. The rolling report-history baseline uses only the preceding 365 days, excluding the current day.

| Q4 2025 Traffy report target | Joint all-target CNN | Urban-only CNN | Rolling report history | Built cover |
| --- | ---: | ---: | ---: | ---: |
| Bang Kapi patch: 3,118 positive cell-days | 32.70% | 32.06% | **35.44%** | 14.58% |
| Nong Chok patch: 107 positive cell-days | **45.79%** | 42.99% | 41.47% | 33.65% |
| **Both patches, 3,225 positives** | 33.14% | 32.43% | **35.64%** | 15.21% |

The joint model beats the urban-only ablation by **0.71 percentage points** overall, but a paired seven-day-block bootstrap gives an exploratory **95% interval from −1.37 to +3.05 points**. It trails rolling report history by **2.50 points** overall; the corresponding interval for joint minus history is **−4.86 to +0.48 points**. These intervals account for some within-week dependence, but 14 blocks and previously inspected Q4 labels limit inference. The incremental benefit of adding satellite targets is therefore **not established**. In Nong Chok alone, the joint model beats rolling history by 4.33 points; Bang Kapi dominates the pooled count and goes the other way.

The earlier satellite-only CNN captured **3.74%** of the same 107 Nong Chok Q4 report-positive cell-days. The new joint model captures **45.79%** there. This large cross-target improvement follows direct training on earlier Traffy reports; the urban-only ablation shows that satellite labels add, at most, a much smaller increment on this check.

For the **13 uniquely matched Q4 BMA incident cell-days** inside Bang Kapi, the joint BMA-site head captures **7/13 (53.85%)** in its top 10%, versus **10/13 (76.92%)** for urban-only. Its conditional depth MAE is **3.52 cm**, worse than a constant 15 cm prediction at **3.08 cm**. These few incidents reuse locations inferred from a 2026 BMA inventory and cannot establish depth generalization. The BMA result gives no evidence of a benefit from joint training.

This is a **retrospective diagnostic, not a forward-time forecast**: the joint model saw 2026 satellite labels, the 2025 daily rain input can include hours after a complaint, OSM/inventory snapshots can postdate events, and Q4 labels were inspected in earlier model development. EOS-RS Bangkok and GISTDA 2026 maps were included in joint fitting, so earlier satellite holdout scores do not apply. Neither head gives calibrated flooding probabilities or verified rainfall counterfactuals.

Reproduce from caches with `.venv/bin/python flood_multi_target_train.py --mix urban_only` and `.venv/bin/python flood_multi_target_q4_diagnostic.py`. The [full metrics](artifacts/flood_open_weak_train/multi_target/q4_2025_urban_diagnostic.json), [urban-only training record](artifacts/flood_open_weak_train/multi_target/urban_only/training_run.json), and [evaluation code](flood_multi_target_q4_diagnostic.py) preserve the comparison.
