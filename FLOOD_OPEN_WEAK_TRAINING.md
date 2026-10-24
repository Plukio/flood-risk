# Expanded open-data flood training

Run 22 October 2026. This run **fits a new checkpoint** using the newly inspected open maps and the existing multi-region GISTDA training maps. It reports data coverage and training behavior only; it does not claim an improvement on a held-out flood outcome.

## Training data

All maps were put on 20 × 20 km patches at 250 m cell spacing. Polygon masks were rasterized at 50 m first and averaged to cell fractions. The sources remain separate in the manifest and sampler.

| Source | Scenes used | Label treatment | Mapped positive area in cells with >20% WorldCover built cover |
| --- | ---: | --- | ---: |
| [UNOSAT Sentinel-1, 2021](https://data.humdata.org/dataset/water-extents-in-central-and-western-parts-thailand-as-of-21-october-2021) | **59 patch-dates**, from four observations of one 2021 flood episode | Flood fraction positive; analyzed non-water **weak likely dry**; outside the analysis footprint unknown. Five scenes are dry-only. | **1.0%** |
| [UNOSAT Chiang Mai, 2024](https://data.humdata.org/dataset/water-extents-from-5-to-10-october-2024-over-chiang-mai-and-lamphun-provinces) | **1** cumulative 6–10 Oct patch | Flood fraction positive; analysis extent minus cloud obstruction weak likely dry. | **68.6%** |
| [EOS-RS Hat Yai, 2025](https://data.humdata.org/dataset/eos-rs-thailand-hat-yai-floods-nov-2025) | **1** patch | Blue pixels positive; transparent pixels unknown. | **42.3%** |
| [EOS-RS Bangkok, 2026](https://data.humdata.org/dataset/eos-rs-thailand-flood-bangkok-sep-2026) | **2** patches: Nong Chok and Bang Kapi core | Blue pixels positive; transparent pixels unknown. | **46.8%** |
| [GISTDA daily maps, 2026](https://data.go.th/dataset/flood-disaster-data) | **142** external-region patch-dates | Mapped polygon fraction positive; outside polygons unknown. | **1.0%** |

**Total: 205 scenes across 24 regions.** A patch-date is a training view, not an independent flood event. The two EOS Bangkok patches share one event, and the 2021 UNOSAT dates belong to one extended episode. The percentages above weight each 250 m cell's 2021 WorldCover class fraction by mapped positive area; they do not verify street-level water.

The scene archive is [data/flood_open_weak_train](data/flood_open_weak_train), with source/date, mapped-water fraction, observed-area fraction, spatial features, and weather vector saved separately for every scene. The [data audit](artifacts/flood_open_weak_train/data_audit.json) records individual areas and catalog links. An integrity check found all 205 arrays finite, fractions within `[0,1]`, UNOSAT positive fractions inside its observed fraction, and zero observed-dry area for EOS-RS/GISTDA.

## Model and fitting

The new [weather-conditioned U-Net](flood_open_weak_train.py) takes 13 spatial channels—SRTM elevation and relative relief, OSM canal/road density and distance, and six WorldCover classes. Nine patch-scale rainfall/season values condition its compressed bottleneck. The [open-data builder](flood_open_weak_data.py) handles the source masks.

The loss asks the model to rank mapped positive area within each patch. UNOSAT alone adds a **small likely-dry penalty** inside its analyzed area. EOS-RS and GISTDA transparency/absence never enters a dry-label loss. Source sampling was set to 30% UNOSAT 2021, 15% UNOSAT 2024, 15% EOS-RS Hat Yai, 20% EOS-RS Bangkok, and 20% GISTDA 2026. Rotations and flips vary spatial training views; they do not create new independent storms.

Training ran **48 epochs and 1,536 gradient steps**. The mean training objective moved from **0.129** in epoch 1 to **−0.534** in epoch 48; its minimum sampled-epoch mean was **−0.772**. This loss includes a constant offset of `log(6,400)` from the 80 × 80 cell density term, so negative values are expected and **do not measure predictive accuracy**. Source sampling changes the mix of examples each epoch. See the [training curve](artifacts/flood_open_weak_train/training_loss.png) and [full run record](artifacts/flood_open_weak_train/training_run.json).

The saved [checkpoint](artifacts/flood_open_weak_train/weather_unet_all_open_weak_sources.pt) includes feature order, normalization, source names, and learned weights. A load-and-forward check produced finite 80 × 80 scores; changing a normalized rain input changed its outputs, confirming that the weather conditioning path is connected. That is a software check, not validation of a rain counterfactual.

## Interpretation limits

- **The checkpoint emits relative mapped-water scores, not flood probabilities.** A 30% EOS blue cell is 30% proxy map coverage, not a 30% probability of flooding.
- The rainfall vectors are retrospective **full local calendar-day** summaries. Some rain may fall after the source satellite pass; exact acquisition hours were not established consistently. Use the checkpoint only for exploratory scenario maps until the event timing is resolved.
- The 2024 Chiang Mai layer is cumulative over several days, and the EOS maps are preliminary radar proxies. Built-up pixels may be missed or misclassified. Repeated augmented views of these few urban scenes can be memorized.
- The OSM snapshot is from 2025 and therefore postdates the 2021 UNOSAT flood. Infrastructure changes since 2021 can enter those spatial features.
- All 2026 GISTDA dates and the 27 Sep Bangkok EOS map were included in fitting. Prior held-out scores cannot be reused as a performance estimate for this checkpoint. The existing checkpoints remain in place.

To rebuild locally:

```bash
.venv/bin/python flood_open_weak_data.py
.venv/bin/python flood_open_weak_train.py
```
