# Does more geography help the Bangkok flood-report CNN?

We expanded the public-data experiment from one year in Bang Kapi and five other districts to **2022–2025** in **eleven districts**. The target is whether a direct [BMA Traffy Fondue](https://data.bangkok.go.th/) flood report's **100 m buffer** touches a 250 m cell on a day. This is a complaint-derived proxy, not verified inundation or water depth. A cell with no report is a zero for that proxy, not a checked dry cell.

## Data and comparison design

The original six districts are Bang Kapi, Chatuchak, Prawet, Suan Luang, Saphan Sung, and Wang Thonglang. The expansion adds Bang Khen, Don Mueang, Bang Khae, Lak Si, and Bang Na. These five add northern, western, and southern coverage while fitting the existing 250 m CNN grid. This is a purposive sample, not all Bangkok districts.

The public [2022](https://data.bangkok.go.th/dataset/traffy-fondue-2565), [2023](https://data.bangkok.go.th/dataset/traffy-fondue-2566), and [2024](https://data.bangkok.go.th/dataset/traffy-fondue-2567) Traffy archives supply **937** direct flood reports in Bang Kapi and **7,678** across the ten other districts after date, category, coordinate, and BMA polygon checks. The [five-district extraction audit](data/flood_bangkok_transfer/traffy_flood_2022_2024_five_districts_audit.json) and [ten-district audit](data/flood_bangkok_transfer/traffy_flood_2022_2024_ten_districts_audit.json) give annual and district counts. The large source archives were streamed and discarded. The retained extracts contain only time, coordinates, district, and flood category for added districts; no text, addresses, photos, or IDs. The [2025 source](https://data.bangkok.go.th/dataset/traffy-fondue-2568) is the same as in the earlier experiment.

Every controlled CNN comparison uses the same architecture, 100 m report buffer, 250 m cells, terrain, nearby BMA pumps/gates, prior report counts, season, and hourly [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api) rain summarized by day. [BMA LiDAR](https://cpudgiapp.bangkok.go.th/image/rest/services/DTM_LiDAR/ImageServer?f=pjson) supplies terrain. There are no OSM or satellite channels in these cross-district CNNs. Historical rain is a district-point gridded estimate, and available pump/gate locations are applied retrospectively to older years. Prior report counts reset each January; 2025 counts stop updating after September so test outcomes cannot enter later test features.

The four-year variants train on 2022–2024 plus January–July 2025. The one-year variants train on January–July 2025. All select a checkpoint and calibrate on **Bang Kapi August–September 2025**, then test on the same **527 Bang Kapi cells during October–December 2025**. The October–December period has already been inspected in earlier work, so this is an **exploratory reused test**.

## Bang Kapi held-out result

Spearman compares each cell's predicted rank with its October–December report-positive-day count. Daily average precision assesses ranking of individual cell-days; the last column checks predicted complaint counts against the **242** observed positive cell-days.

| Training data or benchmark | Training positive cell-days | Cell-rank Spearman | Pairwise agreement | Daily AP | Predicted vs observed cell-days |
|---|---:|---:|---:|---:|---:|
| Historical count, Jan–Sep 2025 | — | 0.490 | 77.7% | — | — |
| Per-cell terrain/OSM/rain tree | 363 | 0.460 | 78.3% | 0.0198 | 209 vs 242 |
| Bang Kapi, 2025 only CNN | 363 | 0.492 | 80.5% | 0.0154 | 436 vs 242 |
| Six districts, 2025 only CNN | 2,265 | 0.513 | 82.1% | 0.0151 | 1,150 vs 242 |
| **Bang Kapi, 2022–2025 CNN** | **2,629** | **0.540** | **83.7%** | **0.0344** | **110 vs 242** |
| Six districts, 2022–2025 CNN | 14,680 | 0.510 | 81.7% | 0.0281 | 143 vs 242 |
| Eleven districts, 2022–2025 CNN | 23,753 | 0.480 | 79.7% | 0.0291 | 118 vs 242 |

For Bang Kapi, **earlier local years help**: the matched local CNN rises from 0.492 to 0.540 Spearman, and daily AP from 0.0154 to 0.0344. Simply pooling other districts' four-year examples does not help this target: six and eleven districts fall to 0.510 and 0.480. The larger models also remain poorly calibrated in count, predicting about half or fewer of the 242 observed report-positive cell-days. Extra geography is useful evidence to study transfer, but naive pooling does not improve this Bang Kapi ranking.

Sampling Bang Kapi on **50% of training steps** in the eleven-district four-year CNN raised its Bang Kapi Spearman from **0.480 to 0.492** and daily AP from **0.0291 to 0.0360**; it predicted **124 vs 242** positive cell-days. This still trails the local-only CNN's **0.540** Spearman. The [local-balance experiment](artifacts/flood_bangkapi_surface/surface_crossdistrict_terrain_cnn_multiyear_local_balance_metrics.json) therefore reduces some dilution from pooled training, without resolving geographic shift or count calibration.

The full [six-district metrics](artifacts/flood_bangkapi_surface/surface_crossdistrict_terrain_cnn_multiyear_metrics.json), [eleven-district metrics](artifacts/flood_bangkapi_surface/surface_crossdistrict_terrain_cnn_multiyear_eleven_metrics.json), [cell comparison table](artifacts/flood_bangkapi_surface/surface_crossdistrict_terrain_cnn_multiyear_eleven_cell_comparison.csv), and [interactive rank map](artifacts/flood_bangkapi_surface/surface_crossdistrict_terrain_cnn_multiyear_eleven_comparison_map.html) allow direct inspection. The [one-year six-district metrics](artifacts/flood_bangkapi_surface/surface_crossdistrict_terrain_cnn_2025_metrics.json) are the reference for the two one-year CNN rows. The per-cell tree uses the same 100 m label and test cells, but has a different model and feature set.

## Transfer test

The separate Bang Khae test excludes Bang Khae labels from training, checkpoint selection, and calibration. Source models train on the ten remaining districts; Bang Kapi August–September selects and calibrates them. Bang Khae January–September report history remains an input feature, and October–December 2025 is the test. This evaluates geographic transfer of the trained CNN, not a zero-information forecast for a district with no local records.

Bang Khae has **840** grid cells and **248** report-positive cell-days in the test. The target district was chosen after checking that it had enough reports for a useful evaluation; this is exploratory, not a blind selection.

| Bang Khae ranking | Training positive cell-days in other districts | Cell-rank Spearman | Pairwise agreement | Daily AP | Predicted vs observed cell-days |
|---|---:|---:|---:|---:|---:|
| Bang Khae historical count, Jan–Sep | — | **0.329** | 66.2% | — | — |
| Ten source districts, 2025 only | 3,103 | 0.313 | 71.6% | 0.0153 | 376 vs 248 |
| Ten source districts, 2022–2025 | 22,293 | 0.324 | **72.4%** | **0.0285** | **41 vs 248** |

Older source years improve daily precision and pairwise ranking in Bang Khae, but the model's cell-rank Spearman still does not beat Bang Khae's simple historical count, and the 2022–2025 model misses most positive-day volume. The [transfer metrics](artifacts/flood_bangkapi_surface/surface_crossdistrict_terrain_cnn_bangkhae_transfer_metrics.json) and [840-cell table](artifacts/flood_bangkapi_surface/surface_crossdistrict_terrain_cnn_bangkhae_transfer_cell_comparison.csv) give the full result. Because checkpoint selection and calibration use Bang Kapi, the Bang Khae calibration gap also shows how poorly one district's score scale can transfer to another.

## Local adaptation and calibration check

We tested two pre-October ways to use Bang Khae data. First, we replaced the Bang Kapi August–September CNN intercept adjustment with one fitted to Bang Khae's August–September report labels. For the four-year CNN, the raw scores summed to **5,335** positive cell-days in that calibration period versus **161** labels, requiring a large correction; the corrected scores then predicted only **58** in October–December versus **248** observed. The one-year CNN's target intercept instead predicted **994**. A target-district intercept alone did not fix the changing score scale across seasons.

Next, a district-specific monotone Poisson rain/season model, trained through July 2025 and level-adjusted on August–September 2025, estimated the expected number of positive cells each day. A common shift to the CNN cell logits made their probabilities sum to that day's estimate, while preserving the within-day CNN order. All fitting and adjustment end before the October–December evaluation; actual test-period rain is an input. The count model estimated **187 vs 248** Bang Khae positive cell-days. A fixed 50/50 logit blend with Bang Khae's January–September cell history was also tested. The test period has been reused in the wider project, so all comparisons are exploratory.

| Bang Khae model, October–December 2025 | Cell Spearman | Daily AP | Brier | Daily count MAE | Predicted vs observed cell-days |
|---|---:|---:|---:|---:|---:|
| Historical Jan–Sep cell rank | 0.329 | — | — | — | — |
| One-year source CNN | 0.313 | 0.0153 | 0.003245 | 4.752 | 376 vs 248 |
| One-year CNN, Bang Khae intercept | 0.312 | 0.0153 | 0.003891 | 9.980 | 994 vs 248 |
| One-year CNN + local rain count | 0.298 | 0.0377 | 0.003142 | 2.914 | 187 vs 248 |
| One-year CNN + local rain count + local cell-history blend | 0.296 | **0.0443** | **0.003137** | 2.914 | 187 vs 248 |
| Four-year source CNN | 0.324 | 0.0285 | 0.003192 | **2.767** | 41 vs 248 |
| Four-year CNN, Bang Khae intercept | 0.324 | 0.0285 | 0.003186 | 2.798 | 58 vs 248 |
| Four-year CNN + local rain count | 0.323 | 0.0401 | 0.003141 | 2.914 | 187 vs 248 |
| Four-year CNN + local rain count + local cell-history blend | 0.317 | 0.0424 | 0.003140 | 2.914 | 187 vs 248 |
| Local cell history + local rain count, without CNN | **0.329** | 0.0428 | 0.003142 | 2.914 | 187 vs 248 |
| Four-year CNN + constant recent daily rate | 0.316 | 0.0079 | 0.003196 | 4.000 | 243 vs 248 |

The local rain count correction improves the four-year CNN's AP, Brier score, log loss, and total count, but **does not improve geographic Spearman**. Its daily count MAE is slightly worse than the source CNN's, so even the stronger total count is not comprehensive calibration. The fixed local-history blend has slightly better AP but weaker geographic rank. Local history without the CNN ranks marginally higher than the CNN and responds to rain only by changing the district-wide daily total; it cannot move risk between cells as rain conditions change. None of these exploratory differences establishes that the transferred CNN generalizes well.

The rain count model also underpredicted earlier October–December blocks after August–September adjustment: **Bang Khae 55 vs 224 in 2023, 89 vs 156 in 2024**, and 187 vs 248 in 2025; **Bang Kapi 41 vs 171, 48 vs 82**, and 137 vs 242, respectively. These 2023–2024 checks refit using only data available before each test block. Their daily count MAE is lower than a constant recent-rate baseline, but the persistent seasonal undercount shows that a simple offset and rain model do not solve temporal shift. The [count-model metrics](artifacts/flood_bangkapi_surface/surface_local_rain_count_calibration_2025_metrics.json), [hybrid metrics](artifacts/flood_bangkapi_surface/surface_bangkhae_spatial_rate_hybrid_metrics.json), and [hybrid 840-cell table](artifacts/flood_bangkapi_surface/surface_bangkhae_spatial_rate_hybrid_cell_comparison.csv) contain the complete results.

## Can partial fine-tuning transfer the CNN?

We pretrained the same CNN on the **ten source districts, 2022–July 2025**, holding Bang Khae out. We then allowed either just its two output heads, or its last spatial convolution plus those heads, to update on Bang Khae labels. We tried both **2022–July 2025** (1,460 positive cell-days) and the more limited **January–July 2025** (95). The first spatial convolution remained frozen. The adaptation epoch was selected by Bang Khae August–September AP; the intercept was fitted on the same August–September block, before evaluating October–December. That block has **161** positive cell-days. We also trained the identical CNN from scratch on Bang Khae through July, with local feature normalization and the same selection/calibration dates.

| Bang Khae model | Selected adaptation epoch | Cell Spearman | Daily AP | Brier | Predicted vs observed cell-days |
|---|---:|---:|---:|---:|---:|
| Source CNN, Bang Khae Aug–Sep intercept | — | 0.324 | **0.0285** | 0.003186 | 58 vs 248 |
| Heads only, all earlier Bang Khae years | **0** | 0.324 | **0.0285** | 0.003186 | 58 vs 248 |
| Last spatial block and heads, all earlier years | **0** | 0.324 | **0.0285** | 0.003186 | 58 vs 248 |
| Heads only, Jan–Jul 2025 | **0** | 0.324 | **0.0285** | 0.003186 | 58 vs 248 |
| Last spatial block and heads, Jan–Jul 2025 | **0** | 0.324 | **0.0285** | 0.003186 | 58 vs 248 |
| Bang Khae-only CNN, 2022–Jul 2025 | — | **0.360** | 0.0274 | **0.003169** | 135 vs 248 |

All four fine-tuning runs reduced August–September AP after their first epoch, so pre-October selection retained the original source weights at epoch zero. This is a negative result for these two partial-update choices, training windows, and learning rates; it does not rule out all adaptation methods. An initial August-only selection block had just **24** positives and also selected epoch zero, so the main comparison uses the larger August–September block. The local-only CNN improves geographic rank but still underpredicts the report count and has slightly lower daily AP. Its local normalization differs from the frozen-source variants, so the result does not isolate weight initialization alone. The [fine-tuning metrics and curves](artifacts/flood_bangkapi_surface/surface_crossdistrict_bangkhae_partial_finetune_augsep_metrics.json) and [cell table](artifacts/flood_bangkapi_surface/surface_crossdistrict_bangkhae_partial_finetune_augsep_cell_comparison.csv) preserve the comparison. This one held-out district and reused test period do not establish performance for a genuinely new Bangkok district.

The [feature shift and ablation report](FLOOD_BANGKOK_FEATURE_ABLATION.md) tests whether absolute elevation, pump/gate inventory, report history, calendar, or rain channels explain the transfer failure. No single removal improves both Bang Khae geographic ranking and calibration; local report history and rain remain important for ranking despite distribution shift.

The [canal-feature experiment](FLOOD_BANGKOK_CANAL_FEATURES.md) adds mapped canal length and nearest-canal distance. It improves the transferred CNN's Bang Khae geographic rank and daily AP, though count calibration remains weak and the local-only geographic rank falls.

## What the expansion means for rain scenarios

The historical-count benchmark cannot vary with rain. The local count model can vary a district's expected report count with rain, while the CNN can vary the spatial scores. Neither changed score is a validated physical flood simulation. The four-year Bang Kapi CNN is currently the strongest of these models for Bang Kapi report ranking, yet it underpredicts the observed positive-day count. The Bang Khae adaptation improves some report-proxy metrics, but still undercounts and does not improve geographic rank. Before using "twice the rain" for decisions, evaluate rain-response direction and calibration on storms and districts not used to select the model, and obtain independently checked wet/dry observations for actual inundation probabilities.

Rebuild the added extracts with `.venv/bin/python flood_bangkok_traffy_history.py --expanded`, then run `.venv/bin/python flood_bangkok_spatial_cnn_multiyear.py` and `.venv/bin/python flood_bangkok_spatial_cnn_multiyear.py --expanded`. Generate the map with `.venv/bin/python flood_bangkok_multiyear_comparison_map.py`. Rebuild the spatial transfer check with `.venv/bin/python flood_bangkok_spatial_cnn_transfer_test.py`, the local count model with `.venv/bin/python flood_bangkok_local_rate_calibration.py`, and the hybrid with `.venv/bin/python flood_bangkok_spatial_rate_hybrid.py`. Run the partial fine-tuning comparison with `.venv/bin/python flood_bangkok_spatial_cnn_partial_finetune.py`.
