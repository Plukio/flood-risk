# Simulation engine for Bangkok spatial pretraining

**Decision note, 23 October 2026.** We have a 10 m Bang Kapi BMA LiDAR DTM,
rainfall scenarios, and mapped 2021 BMA pipe/canal geometry. Pipe-to-inlet
links, operating outlets, gate/pump state, and time-matched boundary levels
are not verified. The intended use is to generate many transparent synthetic
flow examples before fitting a neural surrogate.

| Engine | What it can do for this project | Current practical limit | Fit |
| --- | --- | --- | --- |
| **[SFINCS](https://sfincs.readthedocs.io/en/latest/)** | Open-source 2D raster flood model with rainfall, infiltration, Manning roughness, barriers, water-level/discharge forcing and NetCDF maps. Recent documentation also describes [pumps, culverts, gates](https://sfincs.readthedocs.io/en/latest/input_structures.html) and simplified [urban-drainage zones](https://sfincs.readthedocs.io/en/latest/parameters.html#urban-drainage). HydroMT-SFINCS helps build model inputs from GIS. | Its urban-drainage zones are capacity abstractions, not verified BMA sewer routing. The newer drainage-zone feature is marked prototype. A matched surface-only trial now runs locally; drainage and observed calibration remain open. | **First choice** for reproducible surface-flow training runs. |
| **[HEC-RAS](https://www.hec.usace.army.mil/software/hec-ras/)** | 1D/2D unsteady hydraulics and stormwater pipe-network modeling. Useful for a later instrumented catchment with surveyed pipes, inlets and operating structures. | More engineering setup; the open BMA GIS alone does not identify every working inlet, outlet or event-time state. | Later detailed reference model. |
| **[EPA SWMM](https://www.epa.gov/water-research/storm-water-management-model-swmm)** | Open-source urban runoff and 1D drain/sewer routing, including storage and pumps. | A pipe network and outfall boundary must be assembled and checked. It does not on its own output a 2D street-water map. | Add or couple after drainage connections are verified. |
| [Our local-inertial prototype](FLOOD_SURFACE_LOCAL_INERTIAL_TEACHER.md) | A small auditable 2D surface solver that already runs on a 10 m LiDAR crop and passes water-balance, lake-at-rest, barrier and time-step checks. | No open boundary, verified drain exchange, or observed calibration yet. | Numerical control for SFINCS setup. |

## Next SFINCS experiments

1. Expand the Bang Kapi 10 m domain with a buffer and open boundaries so the
   area of interest is away from artificial edges. Check grid and timestep
   sensitivity. Bracket Manning roughness and infiltration assumptions.
2. Run timed rain pulses and record rain, boundary exchange, infiltration and
   final storage. Export depth, arrival time, duration and maximum depth from
   the time-indexed NetCDF output.
3. Represent BMA pipe and pump effects as **bounded drainage-capacity
   scenarios**, including zero drainage. Add mapped structures individually
   only when their attachments and operation can be justified. Treat canal
   water levels as scenario boundaries until time-matched levels are available.
4. Compare these scenarios with observed wet and dry locations and, where
   possible, measured depth. Then train a surrogate on varied storms and
   states while holding out places and storm types. Keep **surrogate-versus-
   simulator skill** separate from **simulator-versus-observation accuracy**.

The current three-seed [cross-city spatial pretraining result](FLOOD_THAI_SPATIAL_PHYSICS_PRETRAIN.md)
uses a 250 m heuristic teacher. A 10 m SFINCS teacher would define surface
movement more concretely. It cannot establish real flood probability, depth,
or doubled-rain impact without independent wet/dry and depth observations.

## Completed matched surface-only trial

On 23 October 2026, the official `deltares/sfincs-cpu:sfincs-v2.4.1` image
(`sha256:cab764e13a6cbac11fff6a6116201bfab3eef33fd9f0892e47a4cacac14b3bf2`)
ran on the same center **96 × 96** crop of the Bang Kapi 10 m DTM as the
[local-inertial solver](flood_surface_local_inertial.py). Inputs were uniform
30 or 60 mm/h for 15 minutes, all cells initially dry, Manning n = 0.035,
zero infiltration, no drains, and closed edges. SFINCS used `advection=0`,
`viscosity=0`, `huthresh=0.0001`, `alpha=0.4`, and `dtmax=2.5` s; its logged
average timestep was 1.0 s. The local solver used a maximum 2.5 s step.
No observed event or water-depth label was used. The date in the SFINCS
config sets the synthetic time axis only.

| Rain rate, 15 min | Rain volume | SFINCS final depth p95 / max | Local final depth p95 / max | Mean / p95 absolute cell difference | Depth correlation |
| --- | ---: | ---: | ---: | ---: | ---: |
| 30 mm/h | 6,912 m³ | 36.87 / 137.27 mm | 38.29 / 141.11 mm | 0.38 / 1.32 mm | 0.99945 |
| 60 mm/h | 13,824 m³ | 71.62 / 295.85 mm | 72.94 / 306.97 mm | 0.52 / 1.69 mm | 0.99967 |

SFINCS final water storage differed from input rain by less than **0.001 m³**
in either case. The two depth maps align to the source coordinates, and
SFINCS's written ground elevations differ by at most **0.0000075 mm** from
the LiDAR values. [Side-by-side maps](artifacts/flood_bangkapi_surface/sfincs_trial/depth_comparison.png)
and the [full numerical comparison](artifacts/flood_bangkapi_surface/sfincs_trial/comparison.json)
are saved with the SFINCS NetCDF outputs and log files. Correlation measures
agreement between two simulations, **not accuracy against real floods**.
The maximum-depth differences are larger than the typical cell differences;
the numerical schemes and wetting rules are not identical.

Reproduce from the repository root with:

```sh
.venv/bin/python flood_sfincs_lidar_trial.py prepare
docker run --rm --platform linux/amd64 -v "$PWD/artifacts/flood_bangkapi_surface/sfincs_trial/rain_30mmh_15min:/work" -w /work deltares/sfincs-cpu:sfincs-v2.4.1 sfincs
docker run --rm --platform linux/amd64 -v "$PWD/artifacts/flood_bangkapi_surface/sfincs_trial/rain_60mmh_15min:/work" -w /work deltares/sfincs-cpu:sfincs-v2.4.1 sfincs
.venv/bin/python flood_sfincs_lidar_trial.py compare
.venv/bin/python flood_sfincs_lidar_trial.py plot
```

This clears a basic implementation check for generating **synthetic**
surface-flow examples. Observational and report-ranking value remain unproven: SFINCS and the local
solver share the same LiDAR and simplified assumptions, so their agreement
does not establish how real Bangkok drains, gates, canals or rainfall affect
flooding. The next useful simulator check is a buffered domain with an open
boundary and bounded drainage scenarios, followed by comparison to observed
wet and dry locations before using its output as a physical teacher.

A subsequent [SFINCS physics pretraining experiment](FLOOD_SFINCS_PHYSICS_PRETRAIN.md)
trained 10 m spatial CNNs on 270 additional SFINCS tile–storm cases. A plain
dilated CNN slightly beat a U-Net and clearly beat a matched local model when
imitating SFINCS on withheld Bang Kapi tiles and an unseen storm. This is
simulator imitation; its benefit to the existing 250 m report-ranking CNN
has not been established.
