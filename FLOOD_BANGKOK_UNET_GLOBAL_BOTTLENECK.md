# U-Net with rain and season at the bottleneck

We tested the proposed architecture on the existing 250 m Bangkok grid. The encoder receives eight spatial channels (terrain, mapped assets, and prior flood-complaint days), two mapped-canal channels, and the valid-cell mask. The nine district-wide values (month sine/cosine and seven rain summaries) are collapsed from the legacy repeated-map cache to a **9-value vector**. A small network turns that vector into learned scale and offset values for the U-Net's compressed feature map (FiLM), before the decoder upsamples it. The rain/season maps do **not** enter the bottleneck model's encoder. The matched `broadcast` U-Net instead reads those nine values as repeated spatial channels.

Both U-Nets use the same 42 fitting districts, eight geographic holdouts, normalized inputs, seed, weighted loss, 900 training batches per epoch, and a maximum of 40 epochs. Bang Kapi August–September 2025 selects the checkpoint; August–September labels in each district fit a local intercept for its test scores. The test is October–December 2025. The label is a **Traffy flood-complaint 100 m buffer touching a 250 m cell on a day**, not measured water on the ground. This test quarter has been reused in earlier experiments, so these comparisons are exploratory.

## Geographic holdout result

Means across the same eight held-out districts; cell Spearman compares the predicted and observed *quarterly number of complaint-positive days per cell*. Daily AP ranks all cell-days within each district.

| Model | Mean cell Spearman | Mean daily AP | Predicted / observed complaint-positive cell-days, total |
|---|---:|---:|---:|
| Historical complaint ranking | **0.318** | — | — |
| Saved canal CNN | 0.316 | 0.02298 | 365 / 974 |
| Broadcast U-Net, selected epoch 22 | 0.309 | **0.02487** | 350 / 974 |
| **Bottleneck U-Net, selected epoch 2** | 0.289 | 0.01488 | 233 / 974 |

The broadcast U-Net gains a small amount of daily AP over the canal CNN but loses cell ranking. The bottleneck model is worse on both measures. It beats historical cell ranking in only two of eight held-out districts and beats canal CNN daily AP in one. Its Bang Kapi selection AP spiked to 0.0502 at epoch 2, then varied widely; that single-district checkpoint choice did not transfer. All three neural models substantially underpredict complaint-positive cell-days even after local intercept calibration.

## Rain-response check

Each scene uses the same 1 October 2025 terrain, canal geometry, and prior complaint history. Only the seven rain summaries change. The index divides each model's **sum of locally adjusted complaint-proxy scores** by its own 20 mm / 2 h scene. It is not flooded area, depth, or a multiplier for affected people.

| Synthetic scene | Broadcast U-Net score index | Bottleneck U-Net score index | Bottleneck cells with lower scores than 20 mm / 2 h |
|---|---:|---:|---:|
| 20 mm / 2 h | 1.00 | 1.00 | 0% |
| 40 mm / 2 h | 1.41 | 1.25 | <0.01% |
| 40 mm / 4 h | 1.72 | 1.43 | <0.01% |
| 20 mm / 2 h, wet prior days | 1.32 | **0.63** | **~100%** |
| 80 mm / 4 h | 2.45 | 1.96 | <0.01% |

The bottleneck's within-day rain response is mostly ordered, but **wetter prior days reduce its scores almost everywhere**. The short U-Net run gave the opposite wet-prior response, so even the response direction is training-sensitive. The spatial order barely moves with 40 mm / 2 h (mean within-district rank Spearman 0.998 against the 20 mm scene). Thus this version mostly rescales existing hotspots; it does not reliably predict which new cells flood under heavier rain. The 80 mm scene exceeds the largest training daily rain total (76.7 mm). Only 3 training district-days jointly matched/exceeded the 40 mm / 2 h summaries, and 21 matched/exceeded the wet-prior scene's seven summaries.

**Decision:** Keep the canal CNN and historical ranking as the retrospective references. Bottleneck conditioning is implemented and useful as an architecture experiment, but these results do not justify adopting it for flood simulation or preparation decisions. Better time-matched rain, drain/canal state, and independently checked wet *and dry* cells are still needed to validate a rain counterfactual.

## Reproduce and inspect

- Run `.venv/bin/python flood_bangkok_unet_global_bottleneck.py --matched-long`.
- [Full metrics and 50-district results](artifacts/flood_bangkapi_surface/surface_crossdistrict_unet_global_bottleneck_v1_matched_long_metrics.json)
- [Learning curves](artifacts/flood_bangkapi_surface/surface_crossdistrict_unet_global_bottleneck_v1_matched_long_learning_curves.png)
- [District comparison](artifacts/flood_bangkapi_surface/surface_crossdistrict_unet_global_bottleneck_v1_matched_long_district_comparison.csv)
- [Scenario response by district](artifacts/flood_bangkapi_surface/surface_crossdistrict_unet_global_bottleneck_v1_matched_long_scenario_district.csv)
