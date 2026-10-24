"""Small conservative 2D local-inertial surface-water solver.

Depth h is metres; face discharge q is m²/s (flow per metre of face width).
Positive qx points right, positive qy points down the raster. Closed edges.
This is a numerical surface-flow teacher, not a calibrated Bangkok flood model.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

G = 9.81


@dataclass
class State:
    depth: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    elapsed_s: float = 0.
    rain_m3: float = 0.
    infiltration_m3: float = 0.


class LocalInertial2D:
    """Finite-volume continuity with a local-inertial face momentum update."""

    def __init__(self, elevation_m: np.ndarray, cell_m: float,
                 manning_n: float | np.ndarray = .035,
                 crest_x_m: np.ndarray | None = None,
                 crest_y_m: np.ndarray | None = None):
        z = np.asarray(elevation_m, np.float64)
        if z.ndim != 2 or min(z.shape) < 2 or not np.isfinite(z).all():
            raise ValueError("Elevation must be a finite 2D raster")
        if cell_m <= 0:
            raise ValueError("Cell size must be positive")
        n = np.broadcast_to(np.asarray(manning_n, np.float64), z.shape).copy()
        if not np.isfinite(n).all() or np.min(n) <= 0:
            raise ValueError("Manning n must be positive and finite")
        self.z = z
        self.dx = float(cell_m)
        self.nx = .5*(n[:, :-1]+n[:, 1:])
        self.ny = .5*(n[:-1, :]+n[1:, :])
        self.cx = np.maximum(z[:, :-1], z[:, 1:]) if crest_x_m is None else np.asarray(crest_x_m, np.float64)
        self.cy = np.maximum(z[:-1, :], z[1:, :]) if crest_y_m is None else np.asarray(crest_y_m, np.float64)
        if (self.cx.shape != (z.shape[0], z.shape[1]-1) or
            self.cy.shape != (z.shape[0]-1, z.shape[1]) or
            not np.isfinite(self.cx).all() or not np.isfinite(self.cy).all() or
            np.any(self.cx < np.maximum(z[:, :-1], z[:, 1:])) or
            np.any(self.cy < np.maximum(z[:-1, :], z[1:, :]))):
            raise ValueError("Face crests must be finite and at least as high as adjacent ground")

    def initial(self, depth_m: float | np.ndarray = 0.) -> State:
        h = np.broadcast_to(np.asarray(depth_m, np.float64), self.z.shape).copy()
        if not np.isfinite(h).all() or np.min(h) < 0:
            raise ValueError("Initial depth must be nonnegative and finite")
        return State(h, np.zeros_like(self.cx), np.zeros_like(self.cy))

    def stable_dt(self, state: State, max_dt_s: float = 5., cfl: float = .4) -> float:
        if max_dt_s <= 0 or not 0 < cfl < 1:
            raise ValueError("Invalid time-step bounds")
        hmax = float(np.max(state.depth))
        velocity = max(float(np.max(np.abs(state.qx))),
                       float(np.max(np.abs(state.qy))))/max(hmax, .001)
        wave = np.sqrt(G*hmax)
        return min(max_dt_s, cfl*self.dx/max(wave+velocity, 1e-6))

    @staticmethod
    def _face_flux(old_q: np.ndarray, eta_a: np.ndarray, eta_b: np.ndarray,
                   crest: np.ndarray, n: np.ndarray, dx: float, dt: float):
        # h_face is water above the higher ground/explicit barrier crest.
        h_face = np.maximum(np.maximum(eta_a, eta_b)-crest, 0.)
        gradient = (eta_b-eta_a)/dx
        # Local-inertial momentum: pressure slope plus Manning friction.
        denominator = 1.+G*dt*n*n*np.abs(old_q)/np.maximum(h_face, 1e-6)**(7./3.)
        q = (old_q-G*dt*h_face*gradient)/denominator
        return np.where(h_face > 1e-5, q, 0.)

    def step(self, state: State, dt_s: float,
             rain_mps: float | np.ndarray = 0.,
             infiltration_mps: float | np.ndarray = 0.) -> State:
        if dt_s <= 0 or not np.isfinite(dt_s):
            raise ValueError("Time step must be positive and finite")
        rain = np.broadcast_to(np.asarray(rain_mps, np.float64), self.z.shape)
        infiltration = np.broadcast_to(np.asarray(infiltration_mps, np.float64), self.z.shape)
        if (not np.isfinite(rain).all() or not np.isfinite(infiltration).all() or
            np.min(rain) < 0 or np.min(infiltration) < 0):
            raise ValueError("Rain and infiltration rates must be nonnegative and finite")
        available = state.depth+rain*dt_s
        infiltrated = np.minimum(available, infiltration*dt_s)
        remaining = available-infiltrated
        eta = self.z+remaining
        qx = self._face_flux(state.qx, eta[:, :-1], eta[:, 1:], self.cx,
                             self.nx, self.dx, dt_s)
        qy = self._face_flux(state.qy, eta[:-1, :], eta[1:, :], self.cy,
                             self.ny, self.dx, dt_s)
        # A single donor-cell limiter rescales all its outgoing faces. This
        # keeps each depth nonnegative without breaking shared-face fluxes.
        outgoing = np.zeros_like(remaining)
        outgoing[:, :-1] += np.maximum(qx, 0)*dt_s/self.dx
        outgoing[:, 1:] += np.maximum(-qx, 0)*dt_s/self.dx
        outgoing[:-1, :] += np.maximum(qy, 0)*dt_s/self.dx
        outgoing[1:, :] += np.maximum(-qy, 0)*dt_s/self.dx
        donor_scale = np.minimum(1., remaining/np.maximum(outgoing, 1e-15))
        qx *= np.where(qx >= 0, donor_scale[:, :-1], donor_scale[:, 1:])
        qy *= np.where(qy >= 0, donor_scale[:-1, :], donor_scale[1:, :])
        next_depth = remaining.copy()
        next_depth[:, :-1] -= qx*dt_s/self.dx
        next_depth[:, 1:] += qx*dt_s/self.dx
        next_depth[:-1, :] -= qy*dt_s/self.dx
        next_depth[1:, :] += qy*dt_s/self.dx
        if np.min(next_depth) < -1e-10 or not np.isfinite(next_depth).all():
            raise ValueError("Unstable or negative depth")
        next_depth = np.maximum(next_depth, 0.)
        area = self.dx*self.dx
        new_state = State(next_depth, qx, qy, state.elapsed_s+dt_s,
                          state.rain_m3+float(rain.sum()*dt_s*area),
                          state.infiltration_m3+float(infiltrated.sum()*area))
        expected = float(state.depth.sum()*area+rain.sum()*dt_s*area-
                         infiltrated.sum()*area)
        actual = float(next_depth.sum()*area)
        if abs(expected-actual) > 1e-7*max(1., expected):
            raise ValueError(f"Step lost water: {expected-actual} m³")
        return new_state

    def run(self, state: State, duration_s: float,
            rain_mps: float | np.ndarray = 0.,
            infiltration_mps: float | np.ndarray = 0.,
            max_dt_s: float = 5.) -> State:
        if duration_s < 0:
            raise ValueError("Duration must be nonnegative")
        target = state.elapsed_s+duration_s
        while state.elapsed_s < target-1e-9:
            dt = min(self.stable_dt(state, max_dt_s), target-state.elapsed_s)
            state = self.step(state, dt, rain_mps, infiltration_mps)
        return state


def self_check() -> dict:
    flat = LocalInertial2D(np.zeros((12, 12)), 10.)
    rainy = flat.run(flat.initial(), 600., rain_mps=.020/3600)
    expected = .020*600/3600
    if np.max(np.abs(rainy.depth-expected)) > 1e-9:
        raise ValueError("Uniform closed-domain rain is not uniform")
    infiltrating = flat.run(flat.initial(), 600., rain_mps=.020/3600,
                            infiltration_mps=.010/3600)
    if (np.max(np.abs(infiltrating.depth-.5*expected)) > 1e-9 or
        abs(infiltrating.rain_m3-infiltrating.infiltration_m3-
            infiltrating.depth.sum()*100) > 1e-8):
        raise ValueError("Rain/infiltration water ledger failed")
    pulse = np.zeros((12, 12)); pulse[6, 6] = .20
    moved = flat.run(flat.initial(pulse), 120.)
    if (not 0 < moved.depth[6, 6] < .20 or
        abs(moved.depth.sum()-.20) > 1e-10):
        raise ValueError("Pulse failed to spread conservatively")
    z = np.tile(np.linspace(0, .1, 12), (12, 1))
    lake = LocalInertial2D(z, 10.)
    rest = lake.run(lake.initial(.20-z), 120.)
    if np.max(np.abs(rest.depth-(.20-z))) > 1e-9:
        raise ValueError("Lake at rest moved")
    high_crest = np.maximum(z[:, :-1], z[:, 1:]).copy()
    high_crest[:, 5] = 1.
    wall = LocalInertial2D(z, 10., crest_x_m=high_crest)
    water = np.zeros_like(z); water[:, :6] = .20
    blocked = wall.run(wall.initial(water), 60.)
    if np.max(blocked.depth[:, 6:]) > 1e-10:
        raise ValueError("Water crossed an un overtopped barrier")
    return {"uniform_rain_depth_mm": round(expected*1000, 6),
            "uniform_rain_after_half_infiltration_mm": round(.5*expected*1000, 6),
            "pulse_volume_error_m3": round(float((moved.depth.sum()-.20)*100), 12),
            "lake_rest_max_error_mm": round(float(np.max(np.abs(rest.depth-(.20-z)))*1000), 12),
            "barrier_leak_max_mm": round(float(np.max(blocked.depth[:, 6:])*1000), 12)}


def lidar_demo(dtm_path: Path, size: int = 96) -> dict:
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(dtm_path) as src:
        row = (src.height-size)//2
        col = (src.width-size)//2
        z = src.read(1, window=Window(col, row, size, size)).astype(np.float64)
        dx = float(src.res[0])
        if src.crs.to_epsg() != 32647 or abs(dx-src.res[1]) > .01:
            raise ValueError("Expected square EPSG:32647 DTM cells")
    solver = LocalInertial2D(z, dx)
    results = {}
    for mm_per_hour in (30., 60.):
        result = solver.run(solver.initial(), 15*60.,
                            rain_mps=mm_per_hour/1000/3600,
                            max_dt_s=2.5)
        depth = result.depth
        results[str(mm_per_hour)] = {
            "rain_m3": round(result.rain_m3, 3),
            "surface_m3": round(float(depth.sum()*dx*dx), 3),
            "water_balance_error_m3": round(float(result.rain_m3-depth.sum()*dx*dx), 9),
            "depth_p50_mm": round(float(np.median(depth)*1000), 3),
            "depth_p95_mm": round(float(np.quantile(depth, .95)*1000), 3),
            "depth_max_mm": round(float(depth.max()*1000), 3)}
        if mm_per_hour == 30.:
            fine = solver.run(solver.initial(), 15*60.,
                              rain_mps=mm_per_hour/1000/3600,
                              max_dt_s=1.25).depth
            difference_mm = np.abs(depth-fine)*1000
            results[str(mm_per_hour)]["time_step_half_mean_difference_mm"] = round(float(difference_mm.mean()), 6)
            results[str(mm_per_hour)]["time_step_half_p95_difference_mm"] = round(float(np.quantile(difference_mm, .95)), 6)
    return {"dtm": str(dtm_path), "cells": [size, size], "cell_m": dx,
            "duration_min": 15, "boundary": "closed",
            "max_dt_s": 2.5, "manning_n": .035,
            "infiltration": "zero", "scenarios": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtm", type=Path,
                        default=Path("data/flood_bangkapi_surface/dtm_bangkapi_10m.tif"))
    parser.add_argument("--audit", type=Path,
                        default=Path("artifacts/flood_bangkapi_surface/local_inertial_teacher_audit.json"))
    args = parser.parse_args()
    result = {"self_check": self_check(), "lidar_demo": lidar_demo(args.dtm)}
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
