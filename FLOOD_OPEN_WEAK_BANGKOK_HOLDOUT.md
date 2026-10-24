# Bangkok geographic holdout for open weak-label models

Run on 22 October 2026. The [all-source checkpoint](artifacts/flood_open_weak_train/weather_unet_all_open_weak_sources.pt) already trained on Bangkok EOS-RS labels, so it cannot be fairly scored on those labels. This test retrained the same architecture and fixed 48-epoch schedule while excluding **both Bangkok patches** (`nong_chok` and `bang_kapi_core`) from fitting and feature normalization. No remaining training patch intersects the BMA boundary. The holdout checkpoint used **202 training patch-dates in 22 regions**: 58 UNOSAT 2021, one UNOSAT 2024, one EOS-RS Hat Yai 2025, and 142 GISTDA 2026. Source sampling weights were renormalized after removing Bangkok EOS-RS.

This is a **geographic holdout**, not a forward-time test: some external GISTDA examples are from 2026. It also tests satellite *mapped-water proxies*, not verified street flooding. Transparent EOS-RS pixels and GISTDA areas without polygons are **unknown**, never confirmed dry. The metric is mapped-positive area captured by selecting the top-ranked 10% of cells; random selection has 10% expected capture. Every model and baseline uses the same candidate cells within each row. Crop fraction, built fraction, and low SRTM elevation require no fitting.

| Bangkok target | U-Net | Crop cover | Built cover | Low elevation |
| --- | ---: | ---: | ---: | ---: |
| EOS-RS 27 Sep, Bang Kapi patch, BMA cells | **13.90%** | 9.35% | 11.24% | 9.48% |
| EOS-RS 27 Sep, Nong Chok patch, BMA cells | 26.28% | **29.64%** | 1.96% | 14.98% |
| EOS-RS two patches, mapped-area weighted | **20.65%** | 20.41% | 6.18% | 12.48% |
| EOS-RS two patches, cells with >20% built cover | **14.71%** | 11.35% | 10.60% | 10.16% |
| GISTDA Nong Chok, eight dates, mapped-area weighted | 36.16% | **43.88%** | 0.24% | 16.35% |

A separate urban-target stress check uses **107 Traffy flood-report-positive cell-days on 24 days in Q4 2025** within the Nong Chok BMA patch. The U-Net captures **3.74%** of these reports in its daily top 10% queue, versus **33.65%** for built cover, **30.39%** for January–September 2025 report history, and **1.87%** for crop cover. Its cell ranking has Spearman **−0.106** against the number of report-positive days. This check applies a model trained partly on 2026 external-region maps backward to 2025, so it diagnoses urban-target mismatch rather than forward-time skill. Traffy complaints are reporting-biased positive sites, not measured flood boundaries or verified dry ground.

## Earlier CNNs on the same Bangkok targets

The earlier 15-patch CNNs also excluded Bangkok from fitting and normalization. Their 22-channel inputs contain the same terrain, OSM, rain, season, and WorldCover variables as the U-Net, but their **training data and loss differ**. The first CNN used external 2026 GISTDA maps; the second added six 2025 GISTDA dates. The U-Net used UNOSAT, EOS-RS, and GISTDA. This compares the existing **trained systems**, not architecture in isolation. All rows below were recomputed with the same Bangkok cells and labels as the U-Net rows above.

| Held-out target: top-10% positive capture | U-Net | CNN, 2026 GISTDA | CNN, plus 2025 GISTDA | Simple baseline |
| --- | ---: | ---: | ---: | ---: |
| EOS-RS two Bangkok patches | **20.65%** | 17.23% | 20.18% | Crop cover: 20.41% |
| EOS-RS, cells with >20% built cover | **14.71%** | 10.36% | 9.81% | Crop cover: 11.35% |
| GISTDA Nong Chok, eight dates | 36.16% | 40.18% | 40.88% | **Crop cover: 43.88%** |
| Traffy Q4 2025 report sites | 3.74% | 0.94% | 4.67% | **Built cover: 33.65%** |

The CNN is stronger on its GISTDA-like target and the U-Net is stronger on the EOS-RS built-cover subset. Neither learned a useful urban report ranking. The [CNN comparison metrics](artifacts/flood_open_weak_train/bangkok_holdout/cnn_comparison.json) and [comparison script](flood_open_weak_cnn_comparison.py) preserve the matched calculations.

## CNN trained on the same expanded dataset

To isolate the architecture choice more closely, a **new CNN** was trained on the *same 202 non-Bangkok patch-dates* as the holdout U-Net: 58 UNOSAT 2021, one UNOSAT 2024, one EOS-RS Hat Yai 2025, and 142 GISTDA 2026. Both runs used the same region split, normalization method, positive/weak-dry loss, source sampling, random seed, 1,536 training updates, and fixed 48-epoch endpoint. The sampled source counts match exactly. The CNN broadcasts nine weather values into spatial channels before three convolutions; the U-Net injects them at its bottleneck. The CNN has **8,265 parameters** versus **56,961** for the U-Net. These are many patch-date views but only a few independent flood episodes.

| Bangkok target: top-10% positive capture | U-Net, expanded data | CNN, same expanded data | Simple baseline |
| --- | ---: | ---: | ---: |
| EOS-RS Bang Kapi patch | **13.90%** | 10.35% | Built cover: 11.24% |
| EOS-RS Nong Chok patch | 26.28% | 26.56% | **Crop cover: 29.64%** |
| EOS-RS both patches, area weighted | **20.65%** | 19.18% | Crop cover: 20.41% |
| EOS-RS both patches, cells with >20% built cover | **14.71%** | 11.40% | Crop cover: 11.35% |
| GISTDA Nong Chok, eight dates | 36.16% | 37.19% | **Crop cover: 43.88%** |
| Traffy Q4 2025 report sites | 3.74% | 3.74% | **Built cover: 33.65%** |

Adding the larger, source-diverse training set to a CNN does **not** resolve the urban-target failure. The same-data CNN is modestly closer to the U-Net than the earlier GISTDA-only CNN on EOS-RS, but neither model beats simple cover on the most relevant independent report-site check. The EOS-RS and GISTDA rows remain positive-only satellite proxies. The Traffy row is a backward-time, reporting-biased stress check.

Artifacts: [same-data CNN checkpoint](artifacts/flood_open_weak_train/bangkok_holdout_cnn/weather_cnn_bangkok_holdout.pt), [training record](artifacts/flood_open_weak_train/bangkok_holdout_cnn/training_run.json), and [Bangkok metrics](artifacts/flood_open_weak_train/bangkok_holdout_cnn/bangkok_holdout_eval.json). Reproduce with `.venv/bin/python flood_open_weak_train.py --holdout-bangkok --model cnn` followed by `.venv/bin/python flood_open_weak_bangkok_holdout_eval.py --model cnn`.

For the EOS-RS urban-cover row, the two patches contain **54.26 km²** of mapped positive area in cells with >20% built cover, mostly in the Bang Kapi patch. For the eight GISTDA dates, the built-cover subset contains only **4.48 km² summed across dates**, versus **226.08 km²** in all BMA cells. Repeated area on different dates is counted repeatedly, so those totals are not unique flood extent. GISTDA's crop-dominated target remains a poor proxy for Bangkok street flooding. The 27 Sep GISTDA map may share Sentinel-1 observations with the EOS-RS map; the rows do not represent independent validation events.

**Reading the result:** The U-Net shows a small advantage over cover-only rankings on the EOS-RS Bangkok proxy, especially in the built-cover subset, but does not beat crop cover on the Nong Chok EOS-RS patch or the eight GISTDA dates. The 20.65% versus 20.41% EOS-RS aggregate difference is too small to establish a useful gain from these two patches. More decisively for a city flood-response use case, it fails the held-out urban report stress check. The current checkpoint should not be used to rank Bangkok street inspections. Positive-only capture cannot measure false alarms, precision, flood probability, depth, or the quality of a doubled-rain scenario. The retrospective rainfall features may include hours after the satellite acquisition.

Artifacts: [holdout checkpoint](artifacts/flood_open_weak_train/bangkok_holdout/weather_unet_bangkok_holdout.pt), [training record](artifacts/flood_open_weak_train/bangkok_holdout/training_run.json), [per-patch and per-date metrics](artifacts/flood_open_weak_train/bangkok_holdout/bangkok_holdout_eval.json), and [evaluation script](flood_open_weak_bangkok_holdout_eval.py).

Reproduce from the existing scene cache:

```bash
.venv/bin/python flood_open_weak_train.py --holdout-bangkok
.venv/bin/python flood_open_weak_bangkok_holdout_eval.py
```
