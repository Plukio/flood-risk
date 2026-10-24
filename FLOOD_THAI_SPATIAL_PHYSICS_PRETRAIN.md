# General spatial pretraining before Bangkok report masking

**Exploratory experiment, 23 October 2026.** We trained a spatial water-transfer
model on **20 separate Thai 20 km × 20 km patches** and held out four entire
patches: Bang Kapi, Nong Chok, Chiang Mai city, and Hat Yai city. None of the
held-out patch footprints overlaps a source patch. This stage used no Traffy
reports, BMA road incidents, satellite water labels, or report-history masks.

## What the model learned

Each training example combines a new synthetic rain field, initial surface
water, and a patch's 250 m SRTM elevation, OSM road/canal features, and 2021
WorldCover built/crop/vegetation fractions. The target is the *next synthetic
water state*: surface water, water transferred to four neighbors, and
infiltration. The teacher's infiltration and mobility rates are **heuristics**.
Its surface transfers are allowed only toward lower hydraulic head. Both the
teacher and network preserve water in a nonnegative surface/pipe/infiltration/
outlet ledger. The network's hard direction and continuity rules hold even
before training; the experiment tests whether its **learned spatial filters**
help imitate the synthetic step on unseen geography.

Road and cover inputs affect the heuristic teacher's rates. Canal geometry is
present as an input, but there is no canal hydraulic relation in this teacher;
this test cannot show that the model learned drainage through a canal. One
source patch in Pathum Thani overlaps 35 km² of the wider Bangkok tile grid.
It contains no report labels, but the later Bangkok report transfer is not a
strictly unseen-static-geography test across every district.

We fitted two matched 900-step models per seed: a five-step spatial CNN and a
local-only version with the same constrained transfer head. An untrained
spatial network and unchanged-water persistence are further controls. We used
three seeds (`0`, `10000`, `20000`) and generated 12 fresh storms per held-out
place per scenario. We also rolled each one-step model forward four steps
without teacher forcing and tested a doubled rain pulse. All target values
in this table are **synthetic teacher surface water**, measured in millimetres
of mean absolute error over teacher-wet cells (>1 mm).

| Held-out place | One step: spatial / local | Four steps: spatial / local | Four steps, 2× rain: spatial / local |
| --- | ---: | ---: | ---: |
| Bang Kapi | 0.790 / 1.194 | 1.607 / 2.305 | 2.765 / 3.996 |
| Nong Chok | 0.596 / 0.931 | 1.234 / 1.773 | 2.251 / 3.302 |
| Chiang Mai city | 0.618 / 0.877 | 1.390 / 1.761 | 2.552 / 3.150 |
| Hat Yai city | 0.654 / 0.857 | 1.550 / 1.836 | 2.580 / 3.052 |
| **Mean across 12 seed–place results** | **0.665 / 0.965** | **1.445 / 1.919** | **2.537 / 3.375** |

The spatial model wins each place in all three seeds, including doubled rain.
For one-step normal-rain examples, the untrained constrained model averages
**1.638 mm** and persistence **2.772 mm**. This separates the learned gain
from the gain supplied by the hard physical rules. Mean synthetic flow error
also falls in each held-out place relative to the local-only model. The
model's whole-patch continuity error is at float precision in the audit.

**Interpretation:** this is evidence of geographical transfer of a *synthetic
spatial water-transfer skill*. It does not establish transfer to real Bangkok
floods. The target rates were chosen by us, SRTM at 250 m misses curbs,
inlets, buildings and pipe hydraulics, and the four-step rollout has no
measured time unit. BMA LiDAR and the Bangkok-only pipe/pump/gate map were
excluded from this common-feature stage so that the same channels exist in
all source and held-out places. The next, separate test transfers shared
spatial filters into the strict history-masked Bangkok report task; its
outcome must be judged on held-out reports against a matched scratch model.

## Reproduce

Run [the pretraining script](flood_thailand_spatial_physics_pretrain.py) with
seed offsets `0`, `10000`, and `20000`. It saves the corresponding
[seed-0 audit](artifacts/flood_open_weak_train/multi_target/citywide/thailand_spatial_physics_pretrain_heldout_places.json)
and model weights beside the seed-suffixed audits and weights. The conservative
[flow model](flood_citywide_conservative_flow.py) contains the exact hard rules
and the heuristic teacher.
