# Using heterogeneous flood targets effectively

22 October 2026. The present open-data CNN and U-Net train on satellite mapped water but capture only **3.74%** of held-out Bangkok flood complaints in a daily top-10% queue. Built cover captures **33.65%**. Their satellite labels and urban reports describe different observable consequences of flooding, so simply adding more mapped-water scenes to one positive-only loss is unlikely to solve the urban task.

The first joint CNN described below has now been fitted; see [joint training and its limits](FLOOD_MULTI_TARGET_TRAINING.md). Its predictive improvement remains untested.

## Separate the outcome from how it was observed

| Evidence | What it establishes | Training use | What remains unknown |
| --- | --- | --- | --- |
| GISTDA and EOS-RS mapped water | Source-specific **positive proxy area** | Positive-area ranking, with source and land-cover context | No polygon or a transparent pixel is not verified dry. Radar visibility differs by crops, buildings, and vegetation. |
| UNOSAT wet polygons plus analysis extent | Mapped wet, **weak likely dry** inside a valid analyzed footprint | Positive loss and low-weight, footprint-masked negative loss; exclude permanent water/clouds | Urban radar misses and acquisition-time uncertainty remain. |
| Traffy flood complaints | Time-stamped, geolocated **reported problem sites** | Train an urban impact/report-ranking head; include reporting exposure | No complaint is not dry. Complaints depend on roads, people, and app use. |
| BMA road incidents | Reported road flooding, sometimes depth/duration | Additional urban positives after location matching; optional coarse severity target | Most rows have named roads rather than exact cells; missing reports do not mean zero depth. |

Use one event-cell record with `event_id`, city, observation interval, source, cell, mapped-positive fraction, analyzed/unknown status, location uncertainty, land cover, and source confidence. Rain features must stop at the observation time. Do not inflate a point report into a confirmed 100 m wet disk. For an ambiguous BMA road name, retain a small set of plausible segments and let the loss match one or more of them.

## Model to try

1. A shared terrain/rain/drainage encoder produces a **hazard ranking** over cells.
2. Source-specific satellite observation heads model where each sensor is likely to *map* that hazard. Limit these heads to sensor metadata and land cover so they cannot memorize an event. EOS/GISTDA positives contribute only positive evidence; UNOSAT's analyzed footprint adds low-confidence likely-dry evidence.
3. A separate **urban impact head** combines the shared representation with roads, built cover, and exposure. Train it on Traffy and matched BMA positives as a ranking of *reported sites*. A reporting-propensity term, using pre-event complaint activity and population/road exposure, should be modeled separately from rain-driven hazard. This head does not output a calibrated flood probability.

Start with the existing three-layer CNN as the shared encoder because it is cheap to diagnose; changing architecture can wait until the targets and loss are correct. Compare four versions on identical splits: urban reports only; satellite pretraining then urban fine-tuning; joint satellite plus urban heads; and the current satellite-only model. The key question is whether satellite training improves urban ranking **over reports-only**, rather than whether the joint model fits its satellite maps.

## Loss and sampling rules

- Compute satellite positive-area losses within source/event strata. Never assign a dry BCE target to GISTDA absence or EOS transparency. The current whole-patch softmax is still a competition with unlabeled cells, even without an explicit dry term; treat its output as *mapped-positive density*, not flood probability.
- Give Traffy/BMA their own report-location ranking loss over plausible, exposed cells on the same event. This learns where a report appears conditional on the event. It cannot infer the number of unreported floods without an observation model or verified dry examples.
- Weight **independent city-storm events**, not augmented patch-dates. In the current Bangkok-holdout CNN, the single Chiang Mai scene was sampled 288 times and the single Hat Yai scene 298 times: **586 of 1,536 updates** came from those two scenes. Cap repeated draws and log effective event counts.
- Stratify satellite batches by built and crop cover, with an explicit built-area loss, so the much larger mapped agricultural area does not dominate the urban objective. Tune task weights only on training-city validation events, never the Bangkok holdout.

## Evaluation that can decide whether this helps

For a **Bangkok operational model**, train on earlier report periods and test later storms and held-out districts. At the same daily top-10% cell budget, beat both built cover (**33.65%**) and pre-test report history (**30.39%**) on the current 107 Q4 2025 Nong Chok report-positive cell-days; report uncertainty by storm, not by correlated cells. Treat this existing Q4 check as a development target if model choices are made after seeing it, and reserve a new untouched period for the final estimate.

Keep separate satellite checks for EOS urban-cover mapped area and GISTDA crop-dominated mapped area. Do not average them into one score. For a **zero-shot city-transfer claim**, hold out an entire city with urban report labels; Bangkok-only report training cannot establish that claim. A strict forward-time 2025 test cannot train on 2026 satellite maps. Scenario maps remain relative rankings until rain timing and independently observed wet/dry extent support calibration.
