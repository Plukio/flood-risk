# A more explicit synthetic surface-flow teacher

**Prototype, 23 October 2026.** The previous 250 m teacher moved an invented
fraction of each cell's water downhill at every step. The new
[local-inertial solver](flood_surface_local_inertial.py) works on the cached
**10 m BMA LiDAR terrain**. It advances water depth and flow across shared
cell faces in seconds. This makes the synthetic target a defined numerical
flow problem whose time-step error and water balance can be checked.

For each face between cells A and B, the solver computes water-surface height
`η = terrain elevation + water depth`. It can pass a face only when water is
above the higher ground or a specified barrier crest. Its discharge per unit
width is updated using the local-inertial pressure slope and Manning friction:

```text
h_face = max(0, max(η_A, η_B) − face_crest)
q_next = (q_old − g Δt h_face (η_B − η_A)/Δx)
         / (1 + g Δt n² |q_old| / max(h_face, ε)^(7/3))
```

The same face discharge leaves one cell and enters its neighbor. A donor-cell
limiter rescales all outgoing faces if a cell lacks enough water, keeping
depths nonnegative while conserving water. Rain adds water each step;
infiltration is an optional bounded sink. The time step is limited by cell
size, wave speed, and estimated velocity. Current boundaries are closed.

## Numerical checks

The [audit](artifacts/flood_bangkapi_surface/local_inertial_teacher_audit.json)
uses a 96 × 96 cell crop from the [Bang Kapi 10 m DTM](data/flood_bangkapi_surface/dtm_bangkapi_10m.tif).

| Check | Result |
| --- | ---: |
| Flat, closed grid; 20 mm/hour rain for 10 minutes | Uniform **3.333 mm** depth |
| Same rain with 10 mm/hour infiltration | Uniform **1.667 mm** depth and conserved water ledger |
| Initial pulse | Spread with zero reported volume error |
| Sloping terrain under a level lake | Zero reported depth change |
| Barrier crest above the water surface | Zero reported leakage |
| 30 mm/hour for 15 minutes on LiDAR; 2.5 s versus 1.25 s maximum step | **0.0203 mm** mean depth difference; **0.0637 mm** at the 95th percentile |
| Same LiDAR crop; 30 and 60 mm/hour for 15 minutes | **6,912 and 13,824 m³** added and retained respectively; water-balance error rounds to zero |

The LiDAR demo's 95th percentile water depth is **38.29 mm** at 30 mm/hour
and **72.94 mm** at 60 mm/hour. These are **closed-box, no-drainage scenarios**,
so they are numerical examples, not Bangkok flood forecasts. At 30 mm/hour,
changing the assumed Manning roughness from `0.025` to `0.060` changes that
95th percentile from about **40.5 to 33.0 mm**. Parameters and missing sinks
matter more than the measured 0.0203 mm time-step difference.

## How to use it for pretraining

1. Sample a 10–30 m terrain patch and a time-varying rain field. Specify
   roughness, infiltration, barriers, boundary water level, and any verified
   drain inlets as explicit scenario inputs.
2. Run the solver over several real time steps. Save **depth, face discharges,
   cumulative infiltration, and a valid-domain mask** at each step. Check the
   water ledger on every sample.
3. Train a neural surrogate to predict the next state and several rollout
   steps. Compare it with the solver on entirely held-out places, storm shapes,
   roughness ranges, and doubled rain. Include conservation error, wet extent,
   flow magnitude/direction, and time-step refinement in the audit.
4. Aggregate 10–30 m depth to 250 m only after the solver run if the Bangkok
   report model still uses 250 m cells. Aggregating terrain before routing
   erases roads, depressions and short barriers.

This prototype models **surface flow only**. The BMA GIS contains pipe,
canal, gate and pump geometry, but inlet links, outlet status, event-time
levels and operation remain unverified. Those should enter as uncertain
scenario inputs or later verified connections. More comparable high-resolution
terrain outside Bangkok is needed before claiming nationwide pretraining with
the same solver. Measured wet/dry extent and timed depths are still needed to
calibrate the physical parameters and judge real-world accuracy.

Run `.venv/bin/python flood_surface_local_inertial.py` to reproduce the toy
checks and Bang Kapi audit.
