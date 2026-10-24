"""Map 2026-10-04 GISTDA mapped water against the saved canal-CNN queue."""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

from flood_bangkok_gistda_grid_backtest import OUTPUT, selection_weights
from flood_bangkok_transfer import load_districts
from flood_bangkapi_surface_grid import GRID_M, grid_spec


def main() -> None:
    district = next(d for d in load_districts() if d["code"] == "1003")
    spec = grid_spec(district["polygon"])
    path = OUTPUT / "gistda_20261004_grid_250m.csv"
    with path.open(encoding="utf-8", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["district_code"] == "1003"]
    score = np.asarray([float(row["canal_cnn_20mm_2h_reference_score"])
                        for row in rows])
    selected = selection_weights(score, 0.10)
    fig, ax = plt.subplots(figsize=(9.5, 7.8), dpi=150)
    for row in rows:
        area = float(row["mapped_wet_area_m2"])
        if area <= 0:
            continue
        x = spec["west"] + int(row["col"]) * GRID_M
        y = spec["north"] - (int(row["row"]) + 1) * GRID_M
        ax.add_patch(Rectangle((x, y), GRID_M, GRID_M,
                               facecolor="#2477c4", edgecolor="none",
                               alpha=min(0.95, 0.20 + 0.75 * area / 62500)))
    for row, weight in zip(rows, selected):
        if weight <= 0:
            continue
        x = spec["west"] + int(row["col"]) * GRID_M
        y = spec["north"] - (int(row["row"]) + 1) * GRID_M
        ax.add_patch(Rectangle((x, y), GRID_M, GRID_M, fill=False,
                               edgecolor="#e75b22", linewidth=0.5,
                               alpha=float(weight)))
    boundary = district["polygon"]
    for polygon in ([boundary] if boundary.geom_type == "Polygon" else boundary.geoms):
        x, y = polygon.exterior.xy
        ax.plot(x, y, color="#223b54", linewidth=0.7)
    ax.set_aspect("equal")
    ax.set_xlim(spec["west"] - 500, spec["east"] + 500)
    ax.set_ylim(spec["south"] - 500, spec["north"] + 500)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Nong Chok, 4 Oct 2026: mapped water vs. top 10% canal-CNN cells",
                 fontsize=11)
    ax.legend(handles=[Patch(facecolor="#2477c4", label="GISTDA mapped-water share of 250 m cell"),
                       Patch(facecolor="none", edgecolor="#e75b22",
                             label="Top 10% saved canal-CNN rank (20 mm / 2 h reference)")],
              loc="lower left", fontsize=8, framealpha=0.92)
    fig.text(0.5, 0.02, "CNN rank uses a fixed 2025 synthetic scene; unmapped cells are unknown, not verified dry.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output = OUTPUT / "gistda_20261004_nong_chok_rank_overlay.png"
    fig.savefig(output)
    print(output)


if __name__ == "__main__":
    main()
