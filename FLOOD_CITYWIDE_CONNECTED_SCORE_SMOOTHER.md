# Connected-path smoothing of the masked CNN report score

**Tested 24 October 2026.** The idea is to let a high masked-CNN score influence another cell through a *plausible path*, including a cell with prior reports in between. This is a second-stage score model for the existing report-site queue. The CNN score is not a probability of flooding, and a connected score is not a verified wet area.

## What was built

For each side-adjacent pair of 250 m candidate cells, make directed source-to-target edges. Starting with masked-CNN scores at first-in-a-year candidate cells, propagate a weighted average for one, two, or three hops. A cell with prior reports can relay a message but does not start one. A separate **score-gated path** variant requires each relay cell to rank in the top half of raw CNN logits that day; relay weight then increases with its rank. This tests the suggestion that an intermediate cell should also look flood-prone, but the logits are uncalibrated report scores, not wet/dry predictions. Blend the propagated score with the original CNN score. Choose hop count and blend strength separately for each graph on **January–September 2025 source districts**; then evaluate only once on the geographically excluded **Q4 2025** districts. The same daily 10% total inspection area and 20% first-in-a-year slot reserve apply to every row.

| Connection rule | Edge weight concept | Q4 first-in-a-year capture, 1,664 positives | Q4 overall capture, 7,209 positives |
| --- | --- | ---: | ---: |
| Saved masked CNN | No added propagation | 13.529% | 44.479% |
| Ordinary paths | Equal side-neighbor weight | **14.263%** | **44.649%** |
| Terrain paths | Uphill changes attenuated more than downhill changes | 13.685% | 44.515% |
| Reversed-terrain control | The uphill/downhill preference reversed | **14.603%** | **44.727%** |
| Terrain + building fraction | Terrain, then less weight through cells with more OSM building footprint area | 13.917% | 44.569% |
| Terrain + building + cover | Above, plus attenuation for WorldCover crop/tree/grass area | 13.642% | 44.505% |
| Ordinary paths, high-score relay required | Equal side-neighbor weights; only high-ranked intermediate cells relay | 13.462% | 44.464% |
| Reversed terrain, high-score relay required | Directional control with the same score gate | 13.268% | 44.419% |
| Terrain + building + cover, high-score relay required | Physical proxies with the same score gate | 13.685% | 44.515% |

The terrain/building/cover graph captures about **10.1 fewer** first-in-a-year positive cell-days than ordinary paths. Its seven-day-block interval versus ordinary is **−1.28 to +0.26 percentage points**; versus the original CNN it is **−1.25 to +1.29 points**. The date-only intervals omit geographic and model-selection uncertainty. All Q4 comparisons are exploratory because Q4 has been inspected repeatedly. The reversed-terrain control outperforming the intended downhill direction further weakens a physical interpretation. Ordinary multi-hop continuity may be useful for this *report* queue, but we cannot credit it to terrain or surface physics.

The best ordinary-path novel capture remains below the project's **15–20%** exploratory target. Its **44.649%** overall capture with reserved novel slots is below the **47.113%** no-reserve fixed-history fold queue on the same Q4 setup. It is not an operational replacement for that history rule.
The ordinary-path gain over the original CNN is **0.73 percentage points**, with a seven-day-block interval of **−0.51 to +1.51 points**, so that gain is also uncertain.

Requiring high-scored relay cells does **not** improve the best ungated report queue. The gated terrain/building/cover graph captures **13.685%** versus **13.462%** for the equally gated ordinary graph; the interval for that difference is **−0.42 to +0.88 points**. Its own result remains below the **14.263%** ungated ordinary-path row. The fixed top-half score gate is a ranking heuristic, not a threshold for actual flooding.

**Validation-count audit:** The older queue helper used an 8-bit dot product to count first-in-a-year reports. On seven January–September fold-days it wrapped at 256, giving **11,924** instead of **13,716** source-fold positives pooled across five folds. The helper now uses a safe sum. Source-arm ordering is unchanged because each arm in a fold shares its daily denominator. The Q4 held-out total was **1,664** under both counting methods, so the table above is unchanged.

## Why the physical rule is still uncertain

- **Terrain:** BMA DTM samples were averaged to 250 m. A cell mean height does not reveal the lowest pass between cells, water-surface height, a wall, a kerb, or a culvert. The edge rule therefore reduces weight for uphill moves; it cannot declare an edge open or closed.
- **Buildings:** The January 2025 OSM footprint layer gives the *fraction of each cell occupied by buildings*. It does not show whether a continuous barrier crosses the cell boundary. Roads and drainage gaps can pass through a built cell.
- **Land cover:** WorldCover 2021 crop/tree/grass fractions are a vegetation proxy, **not measured soil infiltration**. Cropland can be waterlogged, and built cover can include floodable roads.
- **Outcome:** Traffy reports observe where people report problems. The absence of a report does not establish a dry cell. The ranking test cannot verify the shape or continuity of a real flood plume.

To test the physical idea as *flood extent*, the next evidence would be event-matched wet/dry observations plus higher-resolution boundary saddles and mapped open/blocked passages. With measured or simulated depth, a flow-connected cell could be constrained through water-surface height rather than by smoothing a report logit.

Reproduce with [the connected-path smoother](flood_citywide_connected_score_smoother.py). The [five-fold result artifact](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_mask_cnn_connected_score_smoother_fivefold.json) includes every source selection, fold-level Q4 result, graph audit and daily captures. The path unit check verifies that a score can cross an intermediate cell with prior reports.
