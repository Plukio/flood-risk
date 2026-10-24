"""Typeset the Bangkok flood-site study as a reproducible academic-style PDF.

Run: .venv/bin/python build_flood_site_academic_paper.py
The manuscript reports exploratory complaint-site ranking, not observed flood
probabilities or monetary savings. Metrics are read from local audit artifacts.
"""

from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, PathPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts/flood_open_weak_train/multi_target/citywide"
OUT = ROOT / "BANGKOK_FLOOD_SITE_RISK_PAPER.pdf"
REPO_URL = "https://github.com/Plukio/flood-risk"
RANK = json.loads((ART / "q4_2025_site_novel_rank_curve.json").read_text())
DISTANCE = json.loads((ART / "q4_2025_site_distance_rank_curves.json").read_text())
TOP5 = next(row for row in RANK["curve"]
            if row["fraction_of_zero_history_cells"] == .05)
with (ART / "q4_2025_site_novel_rank_curve_events.csv").open(
        encoding="utf-8", newline="") as source:
    SITE_EVENTS = list(csv.DictReader(source))
if len(SITE_EVENTS) != 393:
    raise ValueError("Economic section expects 393 held-out site events")
MODEL_ONLY = sum(max(0., float(row["cnn_0.05"])-float(row["history_0.05"]))
                 for row in SITE_EVENTS)
HISTORY_ONLY = sum(max(0., float(row["history_0.05"])-float(row["cnn_0.05"]))
                   for row in SITE_EVENTS)
NET_EXTRA = float(TOP5["cnn_hits"])-float(TOP5["history_hits"])
if abs(MODEL_ONLY-HISTORY_ONLY-NET_EXTRA) > .001:
    raise ValueError("Top-5% site-switch accounting does not reconcile")
BUILDING = np.load(ART / "osm_2025_building_footprint_citywide_250m.npz")[
    "building_fraction"]
DENSE_FOOTPRINT_CUTOFF = .20
DENSE_MODEL_ONLY = 0.
DENSE_HISTORY_ONLY = 0.
for row in SITE_EVENTS:
    if BUILDING[int(row["row"]), int(row["col"])] >= DENSE_FOOTPRINT_CUTOFF:
        DENSE_MODEL_ONLY += max(0., float(row["cnn_0.05"])
                                - float(row["history_0.05"]))
        DENSE_HISTORY_ONLY += max(0., float(row["history_0.05"])
                                  - float(row["cnn_0.05"]))
ORDINARY_LOSS_M = .30
DENSE_LOSS_M = 12.0
BASE_EXPOSURE_DIFFERENCE_M = (
    (Decimal(str(TOP5["cnn_hits"])) - Decimal(str(TOP5["history_hits"])))
    * Decimal(str(ORDINARY_LOSS_M)))
ADDED_EXPOSURE_M = ((MODEL_ONLY-DENSE_MODEL_ONLY)*ORDINARY_LOSS_M
                    + DENSE_MODEL_ONLY*DENSE_LOSS_M)
DISPLACED_EXPOSURE_M = ((HISTORY_ONLY-DENSE_HISTORY_ONLY)*ORDINARY_LOSS_M
                        + DENSE_HISTORY_ONLY*DENSE_LOSS_M)
EXPOSURE_DIFFERENCE_M = ADDED_EXPOSURE_M-DISPLACED_EXPOSURE_M

W, H = 8.27, 11.69  # A4 in inches
LEFT, RIGHT = .76, 7.51
WIDTH = RIGHT - LEFT
BLACK = "#171D24"
BLUE = "#1E3B57"
ACCENT = "#276D78"
MID = "#50606E"
RULE = "#BAC6CC"
LIGHT = "#E6ECEE"
WARM = "#8A4F42"
plt.rcParams.update({"font.family": "DejaVu Serif", "pdf.fonttype": 42})


def new_page(pdf, n, short_title):
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(0, H))
    ax.axis("off")
    if n > 1:
        ax.text(LEFT, H - .43, short_title.upper(), size=7.5, color=MID,
                family="DejaVu Sans", va="top")
        ax.text(RIGHT, H - .43, "BANGKOK FLOOD TRIAGE PROTOTYPE", size=7.5,
                color=MID, family="DejaVu Sans", ha="right", va="top")
        ax.plot([LEFT, RIGHT], [H - .63, H - .63], color=RULE, lw=.65)
    ax.plot([LEFT, RIGHT], [.56, .56], color=RULE, lw=.65)
    ax.text(LEFT, .34, "Working paper and hackathon concept  |  25 October 2026", size=7.2,
            color=MID, family="DejaVu Sans", va="center")
    ax.text(RIGHT, .34, str(n), size=7.5, color=MID,
            family="DejaVu Sans", ha="right", va="center")
    return fig, ax


def finish(pdf, fig, lowest=None):
    if lowest is not None and lowest < .78:
        raise RuntimeError(f"Page content approaches footer: {lowest:.2f} in")
    pdf.savefig(fig, dpi=190)
    plt.close(fig)


_MEASURE = TextToPath()


def text_width(s, size, style="normal"):
    """Rendered width of s in inches, from the actual font metrics."""
    prop = FontProperties(family="DejaVu Serif", size=size, style=style)
    return _MEASURE.get_text_width_height_descent(s, prop, False)[0] / 72


def wrap(text, width=WIDTH, size=10.3, style="normal"):
    """Greedy line breaking against measured width; returns lists of words."""
    lines, cur = [], []
    for word in text.split():
        trial = cur + [word]
        if cur and text_width(" ".join(trial), size, style) > width:
            lines.append(cur)
            cur = [word]
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def para(ax, text, y, x=LEFT, width=WIDTH, size=10.3, lead=.225,
         color=BLACK, after=.12, style="normal", justify=True):
    """Draw a justified paragraph; the last line stays at natural spacing."""
    lines = wrap(text, width, size, style)
    space = text_width(" ", size, style)
    for k, words in enumerate(lines):
        last = k == len(lines) - 1
        natural = sum(text_width(w, size, style) for w in words)
        loose = len(words) > 1 and (width - natural) / (len(words) - 1) > 3 * space
        if last or not justify or len(words) == 1 or loose:
            ax.text(x, y, " ".join(words), size=size, color=color, va="top",
                    style=style)
        else:
            gap = (width - natural) / (len(words) - 1)
            cx = x
            for w in words:
                ax.text(cx, y, w, size=size, color=color, va="top", style=style)
                cx += text_width(w, size, style) + gap
        y -= lead
    return y - after


def section(ax, number, name, y, size=12.0):
    ax.text(LEFT, y, f"{number}  {name}", size=size, color=BLUE,
            weight="bold", va="top")
    return y - .34


def subsection(ax, number, name, y):
    ax.text(LEFT, y, f"{number}  {name}", size=10.1, color=BLUE,
            weight="bold", va="top")
    return y - .27


def fig_caption(ax, number, text, y, width=WIDTH, after=.18):
    return para(ax, f"Figure {number}. {text}", y, width=width, size=8.8,
                lead=.185, color=MID, after=after)


def table_caption(ax, number, text, y):
    return para(ax, f"Table {number}. {text}", y, size=8.8, lead=.185,
                color=MID, after=.10)


def arrow(ax, a, b, color=ACCENT):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                 lw=1.05, color=color))


def cell_text(ax, x, y, s, size=8.5, color=BLACK, bold=False, ha="left"):
    ax.text(x, y, s, size=size, color=color, weight="bold" if bold else "normal",
            ha=ha, va="center")


def layer_thumbnail(ax, x, top, kind):
    """Draw an explicitly schematic four-by-four map layer."""
    pale = "#EEF3F4"
    color_sets = {
        "reports": [[pale, pale, pale, pale],
                    [pale, "#9BC2C5", pale, pale],
                    [pale, ACCENT, pale, "#9BC2C5"],
                    [pale, pale, pale, pale]],
        "terrain": [["#DDEBE9", "#DDEBE9", "#BED6D3", "#BED6D3"],
                    ["#DDEBE9", "#BED6D3", "#94B9B6", "#94B9B6"],
                    ["#BED6D3", "#94B9B6", "#689997", "#689997"],
                    ["#94B9B6", "#689997", "#437A78", "#437A78"]],
        "network": [[pale] * 4 for _ in range(4)],
        "cover": [["#B6CFA7", "#B6CFA7", "#C5C4BC", "#C5C4BC"],
                  ["#B6CFA7", "#C5C4BC", "#C5C4BC", "#B6CFA7"],
                  ["#C5C4BC", "#C5C4BC", "#B6CFA7", "#B6CFA7"],
                  ["#C5C4BC", "#B6CFA7", "#B6CFA7", "#B6CFA7"]],
        "priority": [[pale, pale, "#B9D5D4", pale],
                     [pale, "#9CC4C5", ACCENT, "#B9D5D4"],
                     [pale, pale, "#9CC4C5", pale],
                     [pale, pale, pale, pale]],
    }
    size = .155
    x0, y0 = x + .25, top - .44
    for row, colors in enumerate(color_sets[kind]):
        for col, color in enumerate(colors):
            ax.add_patch(Rectangle((x0 + col*size, y0 - row*size - size),
                                   size, size, facecolor=color,
                                   edgecolor="white", lw=.5))
    if kind == "network":
        road = MplPath(
            [(x0 + .01, y0 - .49), (x0 + .17, y0 - .50),
             (x0 + .18, y0 - .30), (x0 + .34, y0 - .35),
             (x0 + .47, y0 - .39), (x0 + .48, y0 - .30),
             (x0 + .62, y0 - .22)],
            [MplPath.MOVETO] + [MplPath.CURVE4] * 6)
        canal = MplPath(
            [(x0 + .12, y0 - .01), (x0 + .01, y0 - .19),
             (x0 + .38, y0 - .22), (x0 + .25, y0 - .35),
             (x0 + .14, y0 - .46), (x0 + .52, y0 - .48),
             (x0 + .50, y0 - .62)],
            [MplPath.MOVETO] + [MplPath.CURVE4] * 6)
        ax.add_patch(PathPatch(road, facecolor="none", edgecolor="#6F7780",
                               lw=2.6, capstyle="round"))
        ax.add_patch(PathPatch(canal, facecolor="none", edgecolor="#5792AA",
                               lw=2.3, capstyle="round"))


def feature_stack(ax, x, bottom, width, height, layers, face,
                  highlighted=(), grid=5):
    """Perspective sketch of aligned 2D feature maps stacked by channel."""
    shear, step_x, step_y = .15, .065, .065

    def point(u, v, layer=0):
        return (x + width*u + shear*v + step_x*layer,
                bottom + height*v + step_y*layer)

    for layer in range(layers-1, -1, -1):
        corners = [point(0, 0, layer), point(1, 0, layer),
                   point(1, 1, layer), point(0, 1, layer)]
        ax.add_patch(Polygon(corners, closed=True,
                             facecolor=face if layer == 0 else "#D9E9EA",
                             edgecolor=ACCENT if layer == 0 else "#7AA6AA",
                             lw=.85 if layer == 0 else .6, zorder=3+layers-layer))
    for row, col in highlighted:
        u0, u1 = col/grid, (col+1)/grid
        v0, v1 = row/grid, (row+1)/grid
        ax.add_patch(Polygon([point(u0, v0), point(u1, v0),
                              point(u1, v1), point(u0, v1)],
                             closed=True, facecolor=ACCENT, edgecolor="none",
                             zorder=10))
    for k in range(1, grid):
        u = k/grid
        v = k/grid
        ax.plot(*zip(point(u, 0), point(u, 1)), color="white",
                lw=.55, zorder=11)
        ax.plot(*zip(point(0, v), point(1, v)), color="white",
                lw=.55, zorder=11)
    ax.add_patch(Polygon([point(0, 0), point(1, 0), point(1, 1),
                          point(0, 1)], closed=True, facecolor="none",
                         edgecolor=ACCENT, lw=.9, zorder=12))


with PdfPages(OUT, metadata={
    "Title": "Toward Earlier Discovery of Urban Flood Risk in Bangkok",
    "Author": "Bangkok flood modeling project",
    "Subject": "Evidence for a Bangkok flood-site triage prototype and BMA collaboration",
}) as pdf:
    # Page 1 — title, abstract, introduction.
    fig, ax = new_page(pdf, 1, "")
    y = H - .76
    ax.text(LEFT, y, "Toward Earlier Discovery of Urban",
            size=17.5, weight="bold", color=BLACK, va="top")
    ax.text(LEFT, y - .38, "Flood Risk in Bangkok", size=17.5,
            weight="bold", color=BLACK, va="top")
    ax.text(LEFT, y - .91,
            "A spatial triage prototype, retrospective evidence, and a path to BMA field validation",
            size=9.2, color=MID, va="top")
    ax.text(LEFT, y - 1.21, "Bangkok flood modeling project  |  Hackathon working paper  |  25 October 2026",
            size=8.6, color=BLUE, va="top")
    ax.plot([LEFT, RIGHT], [y - 1.49, y - 1.49], color=BLACK, lw=.8)
    y -= 1.83
    ax.text(LEFT, y, "Abstract", size=10.4, color=BLUE, weight="bold", va="top")
    y -= .26
    y = para(ax, "Urban flood preparedness requires choosing where to inspect "
             "before storms, yet city teams cannot survey every street. Past "
             "complaints identify recurring trouble spots but offer less guidance "
             "for places with no recent report. We present a daily 250 m priority "
             "map for Bangkok that combines report history with terrain, roads, "
             "canals, and land cover. A site-trained convolutional neural network "
             "learns a correction to a strong history ranking. We tested the "
             "rankings against first-in-year flood-report sites recorded in "
             "Traffy during Q4 2025 across "
             "districts held out from fitting and model selection. Among 393 new "
             "sites in squares with no prior-year report, screening the top 5% "
             "captured 14.0% of sites versus 6.8% for history alone; mean rank "
             "percentile rose from 60.2 to 65.7. When all district squares "
             "competed for a top-5% shortlist, both methods captured 1.27% of "
             "emerging sites. The gain suggests a way to direct "
             "attention toward overlooked locations, while field validation and "
             "a useful full-district inspection allocation remain open tasks.", y,
             size=9.8, lead=.205, after=.18)
    ax.text(LEFT, y, "Keywords", size=8.5, color=BLUE, weight="bold", va="top")
    ax.text(LEFT + .77, y, "urban flooding; spatial ranking; emerging sites; "
            "field triage; Bangkok", size=8.4, color=MID, va="top")
    y -= .39
    y = section(ax, "1", "Introduction", y)
    y = para(ax, "BMA's road-flood records include a November 2025 event with "
             "waterlogging lasting several hours on Ratchadaphisek Road [1]. "
             "Its FY2026 drainage budget and annual action plan put substantial "
             "resources and explicit targets behind pipes, water control, canals, "
             "and identified flood-risk points [2,3]. A map shortlist could "
             "help decide which places warrant closer attention as that work "
             "continues. Ranking only past complaints may repeatedly select "
             "familiar sites while overlooking emerging ones.", y)
    y = para(ax, "The ranking question is concrete: as a district reviews more "
             "map squares, which new trouble spots enter the list first? "
             "Past reports are a useful starting point [4]. "
             "We test whether the proposed model ranks newly reported "
             "locations higher at equal numbers of unfamiliar squares, then identify "
             "what BMA would need to check before using it in field decisions.", y)
    callout_top = y - .14
    ax.add_patch(Rectangle((LEFT, callout_top - 1.80), WIDTH, 1.80,
                           facecolor="#EDF3F4", edgecolor=RULE, lw=.7))
    ax.text(LEFT + .17, callout_top - .15,
            "2× NEW-SITE CAPTURE AT THE SAME 5% UNFAMILIAR-AREA COVERAGE",
            size=7.7, family="DejaVu Sans", color=ACCENT,
            weight="bold", va="top")
    ax.text(LEFT + .17, callout_top - .38,
            "Both methods rank the same 250 m squares with no report in the prior year.",
            size=8.3, color=BLACK, va="top")
    ax.text(LEFT + .17, callout_top - .58,
            "Of 393 later new report sites, share shortlisted:",
            size=8.3, color=BLACK, va="top")
    for row, (name, value, color) in enumerate((("Report history", 6.819, "#83929D"),
                                                ("Proposed model", 13.957, ACCENT))):
        cy = callout_top - .91 - row * .30
        ax.text(LEFT + .17, cy, name, size=8.7, color=BLUE,
                va="center", weight="bold")
        ax.add_patch(Rectangle((LEFT + 1.90, cy - .085), 3.38 * value / 15, .17,
                               facecolor=color, edgecolor="none"))
        ax.text(LEFT + 5.52, cy, f"{value:.1f}%", size=10.2,
                color=BLUE, weight="bold", va="center")
    ax.plot([LEFT + .17, RIGHT - .17], [callout_top - 1.35] * 2,
            color=RULE, lw=.55)
    ax.text(LEFT + .17, callout_top - 1.46,
            "Proposed model: history score + CNN using report maps, terrain, roads, canals, and cover.",
            size=7.9, color=MID, va="center")
    ax.text(LEFT + .17, callout_top - 1.65,
            "Full district top 5% of all cells: both find 1.27% of 471 new sites.",
            size=7.9, color=MID, va="center")
    y = callout_top - 1.95
    finish(pdf, fig, y)

    # Page 2 — outcome, source data, design.
    fig, ax = new_page(pdf, 2, "Data and study design")
    y = H - .93
    y = section(ax, "2", "Study area, data, and target", y)
    y = para(ax, "Bangkok's 250 m cells combine dated Traffy reports, SRTM terrain, "
             "OpenStreetMap roads and canals, and ESA WorldCover 2021 [5-7]. "
             "These predict reported sites, not observed water. The convolutional "
             "neural network (CNN) has no rain input.", y)
    layer_top = y - .11
    ax.text(LEFT, layer_top, "PAST SIGNAL + PHYSICAL MAP LAYERS",
            size=7.7, family="DejaVu Sans", color=ACCENT,
            weight="bold", va="top")
    ax.text(6.08, layer_top, "MODEL OUTPUT", size=7.7,
            family="DejaVu Sans", color=ACCENT,
            weight="bold", va="top")
    card_top = layer_top - .23
    cards = [(LEFT, 1.12, "Reports (4)", "reports"),
             (2.02, 1.12, "Terrain (3)", "terrain"),
             (3.28, 1.12, "Roads/canals (4)", "network"),
             (4.54, 1.12, "Cover (3)", "cover"),
             (6.08, 1.43, "Priority map", "priority")]
    for x, w, name, kind in cards:
        ax.add_patch(Rectangle((x, card_top - 1.16), w, 1.16,
                               facecolor="#FAFBFB", edgecolor=RULE, lw=.65))
        ax.text(x + w/2, card_top - .18, name, size=7.7,
                color=BLUE, weight="bold", ha="center", va="center")
        layer_thumbnail(ax, x, card_top, kind)
    ax.text(5.90, card_top - .43, "CNN", size=6.4, color=ACCENT,
            family="DejaVu Sans", weight="bold", ha="center", va="center")
    arrow(ax, (5.76, card_top - .68), (6.04, card_top - .68))
    y = fig_caption(ax, 1, "Schematic 250 m grids: four report, three terrain, "
                    "four road/canal, three cover, and two mask channels form 16 "
                    "CNN inputs. Colors illustrate ranking, not observed water.",
                    card_top - 1.31)
    y = subsection(ax, "2.1", "Defining a first-in-year reported site", y)
    y = para(ax, "Most of the 2,326 Q4 site events revisit familiar places. "
             "A model trained to find unfamiliar places needs a separate test "
             "on new sites. We therefore define first-in-year report-site events, "
             "then compare rankings among cells with no own-cell report history.",
             y, after=.08)
    y = para(ax, "Nearby same-day report points form complete-linkage groups; a "
             "medoid selects one evaluation cell. A group is 'new' if none of "
             "its points lies within 250 m of an earlier report in the preceding "
             "365 days [8]. Counting each group once avoids complaint-buffer "
             "spillover into adjacent cells. Q4 has 471 new site events; 393 "
             "have no own-cell history.", y)
    y = subsection(ax, "2.2", "Ranking evaluation and outcome", y)
    y = para(ax, "Before rain, crews cannot check every area; the proposed value "
             "is deciding which unfamiliar places to examine first. We rank "
             "all cells with no own-cell report in the previous 365 days within "
             "each district and day. Of 471 Q4 first-in-year site events, 393 "
             "fall in this candidate set. We ask how highly each later site "
             "medoid ranks, and show capture as the review fraction grows. "
             "No share of staff time is assigned to discovery by this test [11].",
             y, after=.08)
    y = para(ax, "The percentile rank is the share of unfamiliar candidates "
             "scoring below a reported site; higher is better. The capture curve "
             "shows the share of those 393 site events in the top fraction of "
             "unfamiliar candidates. Section 5.3 checks a full district queue "
             "without reserved slots. A practical field test would set "
             "stops from visit logs and measure verified findings per crew-hour. "
             "Training uses earlier site medoids and unlabeled same-day rivals "
             "(Section 4); prevented floods are not measured.", y)
    design_top = y
    ax.text(LEFT, design_top, "HOW THE RETROSPECTIVE TEST IS SEPARATED",
            size=7.7, family="DejaVu Sans", color=ACCENT,
            weight="bold", va="top")
    col_x = (2.35, 4.20, 6.05)
    headers = ("2023-2024", "Jan-Sep 2025", "Oct-Dec 2025")
    for x, name in zip(col_x, headers):
        ax.text(x + .70, design_top - .22, name, size=7.7,
                color=BLUE, ha="center", va="center", weight="bold")
    for row, (name, cells) in enumerate((
            ("40 source districts", ("Train model", "Choose settings", "--")),
            ("10 held-out districts", ("No fitting", "No selection", "Test new sites")))):
        cy = design_top - .47 - row * .29
        ax.text(LEFT, cy, name, size=7.7, color=MID,
                weight="bold", va="center")
        for x, label in zip(col_x, cells):
            ax.add_patch(Rectangle((x, cy - .13), 1.40, .26,
                                   facecolor="#DCEBEC" if label in ("Train model", "Test new sites")
                                   else "#F4F6F7", edgecolor=RULE, lw=.55))
            ax.text(x + .70, cy, label, size=7.3, color=BLUE,
                    ha="center", va="center")
    y = fig_caption(ax, 2, "Five district folds; held-out outcomes do not fit "
                    "or select the model.", design_top - .94, after=.02)
    finish(pdf, fig, y)

    # Page 3 — model architecture.
    fig, ax = new_page(pdf, 3, "Spatial model")
    y = H - .93
    y = section(ax, "3", "Spatial ranking model", y)
    y = para(ax, "For candidate cell i and day t, a 15 x 15 patch X(i,t) contains "
             "16 aligned spatial channels. Each patch pixel represents a 250 m cell. "
             "History maps are built strictly from dates before t; physical channels "
             "are static snapshots. Nonlinear spatial mixing permits conditional "
             "effects such as neighboring report density interacting with low "
             "terrain, canal proximity, and built cover. It does not by itself "
             "identify a hydraulic flow path.", y)
    top = y - .11
    bottom = top - 1.13
    feature_stack(ax, 1.03, bottom, .70, .68, 4, "#E3F0F0",
                  highlighted=((1, 1), (2, 1), (2, 3), (3, 2)))
    feature_stack(ax, 2.73, bottom, .70, .68, 3, "#C9E1E2",
                  highlighted=((1, 2), (2, 2), (3, 2)))
    feature_stack(ax, 4.37, bottom, .70, .68, 3, "#D7E8E5",
                  highlighted=tuple((r, c) for r in range(1, 4)
                                    for c in range(1, 4)))
    feature_stack(ax, 6.56, bottom, .64, .68, 1, "#E6F0EF",
                  highlighted=((2, 2),))
    arrow_y = bottom + .39
    for start, end in ((1.96, 2.58), (3.64, 4.21), (5.29, 6.40)):
        arrow(ax, (start, arrow_y), (end, arrow_y))
    for x, label in ((2.27, "1 x 1"), (3.93, "3 x 3"), (5.83, "1 x 1")):
        ax.text(x, arrow_y + .17, label, size=7.5, color=MID,
                family="DejaVu Sans", ha="center", va="center")
    for x, label in ((1.48, "16 input layers"), (3.18, "8 learned maps"),
                     (4.82, "5 spatial steps"), (6.94, "center score")):
        ax.text(x, bottom - .20, label, size=7.7, color=BLUE,
                family="DejaVu Sans", ha="center", va="center")
    y = fig_caption(ax, 3, "Perspective view of the CNN. Stacks show channels, "
                    "not physical height; grids are simplified from the 15 x 15 "
                    "input. The highlighted center is the score correction.",
                    bottom - .43)
    y = subsection(ax, "3.1", "How the CNN reads a neighborhood", y)
    y = para(ax, "The 15 x 15 input patch covers about 3.75 km across. The entry "
             "layer mixes the 16 values at each cell into eight learned maps. "
             "Five spatial steps scan neighboring cells and mix those maps; "
             "the last two use wider spacing. The center output corrects the "
             "nearby-history score. This learns associations with later reports, "
             "not a simulated water route.", y)
    y = subsection(ax, "3.2", "History reference and residual score", y)
    y = para(ax, "The unfamiliar-cell reference score is a weighted combination of "
             "log-transformed prior-year reports smoothed at approximately 250, "
             "750, and 1,500 m: weights 0.6, 0.3, and 0.1, respectively [9]. "
             "The site CNN starts with a zero-initialized output head and learns "
             "a correction to that score. The blend and training arm are chosen "
             "on source-district mean site rank, without an inspection cutoff.", y)
    ax.text(LEFT + .25, y - .03,
            r"$s_{i,t}=(1-\alpha)\,\mathrm{rank}(h_{i,t})+\alpha\,\mathrm{rank}(h_{i,t}+f_{\theta}(X_{i,t}))$",
            size=12.0, color=BLUE, va="top")
    y -= .52
    y = para(ax, "The small residual design sets a demanding test: the CNN must "
             "contribute spatial information beyond a sensible hand-built prior "
             "while preserving the reference where it is useful. Static features "
             "are normalized using source districts only. A valid-patch mask keeps "
             "scores consistent at the grid boundary.", y)
    y = subsection(ax, "3.3", "Why this is a ranking problem", y)
    y = para(ax, "The data lack verified dry observations. Training a calibrated "
             "flood/no-flood classifier would therefore mislabel unknown cells. "
             "Instead, the model learns whether a known positive reported site "
             "should outrank plausible unreported alternatives within the same "
             "district and day. That objective matches the ranking evaluation, "
             "but its score has no direct probabilistic interpretation.", y)
    finish(pdf, fig, y)

    # Page 4 — training and mask.
    fig, ax = new_page(pdf, 4, "Training and masking")
    y = H - .93
    y = section(ax, "4", "Training design and spatial masking", y)
    y = para(ax, "The earlier cell-trained CNN used report-positive cell-days and "
             "randomly suppressed 10% of neighboring history pixels. A later "
             "synthetic-center variant added repeat sites whose center history was "
             "hidden, sometimes together with a random 3 x 3 block. Source-selected "
             "cell-day gains were small and uncertain [9,10]. The present experiment "
             "changes the positive target to raw-point first-in-year site medoids. "
             "This target redesign and the training comparison both matter; the "
             "gain cannot be assigned to masking alone.", y)
    y = subsection(ax, "4.1", "Two site-trained arms", y)
    y = para(ax, "The new-site-only arm trains on genuine 2023-2024 first-in-year "
             "medoids. Those positive cells naturally have zero own-cell history, "
             "so no synthetic center mask is applied. The group-mask arm adds "
             "repeat-site medoids and zeros their 3 x 3 neighborhood in all four "
             "report-history planes. Physical and geometry channels remain visible, "
             "and these synthetic positives receive 0.3 loss weight. No artificial "
             "mask is used at inference [8].", y)
    top = y - .40
    side = .35
    for gx, label, masked in [(1.24, "Before: repeat-site history", False),
                               (4.69, "During group-mask training", True)]:
        ax.text(gx + .875, top + .18, label, size=8.1, color=MID,
                family="DejaVu Sans", ha="center", va="bottom")
        for rr in range(5):
            for cc in range(5):
                inside = 1 <= rr <= 3 and 1 <= cc <= 3
                fill = "#FFFFFF" if (masked and inside) else \
                       ("#4D8491" if (rr + cc) % 3 == 0 else "#BCD4D8")
                if rr == 2 and cc == 2 and not masked:
                    fill = WARM
                ax.add_patch(Rectangle((gx + cc*side, top - (rr+1)*side),
                                       side - .018, side - .018,
                                       facecolor=fill, edgecolor=RULE, lw=.25))
    arrow(ax, (3.51, top - .87), (4.46, top - .87))
    y = fig_caption(ax, 4, "Schematic 5 x 5 crop from each 15 x 15 report-history patch. "
                    "The optional arm hides the centered 3 x 3 history block at "
                    "repeat sites. Static terrain, road, canal, and cover channels "
                    "are unchanged.", top - 1.99)
    y = subsection(ax, "4.2", "Pairwise loss and comparison sampling", y)
    y = para(ax, "For each positive medoid, comparison cells are unreported, have "
             "zero own-cell history, and are in the same district on the same day. "
             "One is drawn from a similar nearby-history score; another is a "
             "high-history rival. The loss increases when a rival outranks a "
             "positive. A small squared-residual term discourages unnecessary "
             "departure from the reference score.", y)
    ax.text(LEFT + .25, y - .01,
            r"$\mathcal{L}=\frac{\sum_{(i,j)}w_i\,\mathrm{softplus}(s_j-s_i)}{\sum_{(i,j)}w_i}+0.0005\,\overline{f_\theta(X)^2}$",
            size=10.8, color=BLUE, va="top")
    y -= .74
    y = para(ax, "Here i is a positive site, j an unlabeled rival, and w_i is "
             "1 for genuine new-site examples or 0.3 for synthetically masked "
             "repeat sites. This is positive-unlabeled ranking, not wet/dry "
             "classification. The model can still exploit the spatial imprint of "
             "earlier reporting; masking cannot prove that it has learned water "
             "movement.", y)
    finish(pdf, fig, y)

    # Page 5 — quota-free evaluation protocol and main results.
    fig, ax = new_page(pdf, 5, "Evaluation and main result")
    y = H - .93
    y = section(ax, "5", "Geographic evaluation of emerging sites", y)
    y = para(ax, "Five geographic folds each exclude 10 Bangkok districts from "
             "weight fitting and model-arm selection. Training uses 2023-2024 "
             "source-district events. January-September 2025 source districts "
             "select the history/CNN blend and training arm by mean new-site "
             "rank percentile, without a shortlist cutoff. "
             "October-December 2025 reports in excluded districts form the Q4 "
             "evaluation [11]. Earlier test-district reports remain available "
             "as pre-day context. Source selection chooses the new-site-only "
             "arm in all five folds.", y)
    y = subsection(ax, "5.1", "Ranking unfamiliar candidates", y)
    chart_top = y - .22
    panel_top = chart_top - .42
    panel_bottom = chart_top - 1.82
    panel_width = 2.02
    panel_left = (1.03, 3.18, 5.33)
    ax.text(LEFT, chart_top + .08, "CAPTURE OF NEW SITES BY PRIOR-REPORT DISTANCE",
            size=7.5, color=MID, family="DejaVu Sans", va="bottom", weight="bold")
    for lx, color, name in ((4.61, "#83929D", "History"),
                            (5.94, ACCENT, "Site CNN")):
        ax.plot([lx, lx+.20], [chart_top+.11, chart_top+.11], color=color, lw=2.0)
        ax.text(lx+.25, chart_top+.11, name, size=7.1, color=BLUE, va="center")
    panel_names = ("250-500 m", "500-750 m", ">750 m")
    for index, (left, group, name) in enumerate(zip(
            panel_left, DISTANCE["groups"][1:], panel_names)):
        right = left + panel_width
        ax.text((left+right)/2, chart_top-.04,
                f"{name}  |  n={group['events']}",
                size=7.8, color=BLUE, ha="center", va="top", weight="bold")
        ax.text((left+right)/2, chart_top-.24,
                f"Mean rank gain {group['rank_lift_points']:+.1f} points",
                size=7.0, color=MID, ha="center", va="top")
        for tick in (0, 40, 80):
            yy = panel_bottom + tick/80*(panel_top-panel_bottom)
            ax.plot([left, right], [yy, yy], color=LIGHT, lw=.55)
            if index == 0:
                ax.text(left-.08, yy, f"{tick}%", size=6.6,
                        color=MID, ha="right", va="center")
        ax.plot([left, left], [panel_bottom, panel_top], color=BLACK, lw=.7)
        ax.plot([left, right], [panel_bottom, panel_bottom], color=BLACK, lw=.7)
        for tick in (0, 20, 50):
            if index > 0 and tick == 0:
                continue
            xx = left + tick/50*panel_width
            ax.text(xx, panel_bottom-.09, f"{tick}%", size=6.7,
                    color=MID, ha="center", va="top")
        curve = group["curve"]
        xx = [left] + [left + p["fraction_of_unfamiliar_cells"]/.5*panel_width
                       for p in curve]
        for key, color in (("history_capture_pct", "#83929D"),
                           ("cnn_capture_pct", ACCENT)):
            yy = [panel_bottom] + [panel_bottom + p[key]/80*
                                   (panel_top-panel_bottom) for p in curve]
            ax.plot(xx, yy, color=color, lw=1.8, marker="o", markersize=2.1)
    ax.text((LEFT+RIGHT)/2, panel_bottom-.29,
            "Top share of unfamiliar cells reviewed (same pool in all panels)",
            size=7.4, color=MID, ha="center", va="top")
    y = fig_caption(ax, 5, "Q4 first-in-year reported-site capture by nearest "
                    "earlier report distance. The three bands partition 393 "
                    "sites; every panel ranks the same zero-own-history cells.",
                    panel_bottom-.48)
    y = para(ax, "Across all 393 eligible new sites, mean rank percentile rises "
             "from 60.17 for nearby history to 65.70 for the source-selected "
             "CNN/history blend: +5.53 points. A paired district bootstrap "
             "interval is +2.75 to +8.00 points [11]. At 5% of unfamiliar "
             "candidates reviewed, capture is 6.82% versus 13.96%; at 10%, "
             "it is 16.13% versus 20.61%. These cutoffs illustrate the curve; "
             "none is a measured crew budget.", y)
    y = subsection(ax, "5.2", "The gain grows with distance from past reports", y)
    y = para(ax, "History ranks new sites well near older reports. The CNN's "
             "mean-rank gain rises with distance: +0.5 points for 271 sites "
             "250-500 m away, +12.3 for 79 sites 500-750 m away, and +25.2 "
             "for 43 sites beyond 750 m [16]. At a 30% unfamiliar-cell screen, "
             "capture beyond 750 m is 4.7% for history versus 37.2% for the "
             "CNN. Figure 5 shows the distance-stratified curves. The pattern suggests "
             "value where report history is sparse; it does not prove water-flow "
             "learning or define a crew budget.", y)
    y = subsection(ax, "5.3", "The full-district shortlist", y)
    y = para(ax, "The 5% test above compares rankings within unfamiliar cells. "
             "In a full district list where all cells compete, both rankers "
             "capture 24.71% of all 2,326 reported sites and 1.27% of 471 "
             "emerging sites at the top 5% of all cells [15]. At the top 10%, "
             "both also tie: 38.32% of all sites and 3.50% of emerging sites. "
             "The conditional gain within unfamiliar cells has not improved "
             "this full queue. Q4 was reused in development, and reports are "
             "not verified flood extent.", y)
    finish(pdf, fig, y)

    # Page 6 — economics, starting with one event and then shortlist switches.
    fig, ax = new_page(pdf, 6, "Loss from flooding and overlooked sites")
    y = H - .93
    y = section(ax, "6", "Economic loss from an overlooked flooded square", y)
    y = para(ax, "A 250 m square covers 62,500 square metres, but its flooded "
             "footprint may be much smaller. One real flood can still impose "
             "repair, interruption, travel, and public-response costs beyond "
             "the visible water. The cost depends on depth, duration, exposed "
             "people and assets, and alternatives for travel. A Traffy report "
             "does not measure those quantities.", y)
    y = subsection(ax, "6.1", "The same flooded area can expose very different value", y)
    y = para(ax, "A quiet side street and a dense commercial square occupy the "
             "same 250 m grid area, yet the latter can contain many premises "
             "and a busy through-route. Table 1 gives two transparent stress "
             "cases. Neither is an estimated Bangkok average or a forecast "
             "for any Q4 reported site.", y)
    y = table_caption(ax, 1, "Hypothetical loss per square; prem=premises, "
                      "shop-h=shop-hours, p-h=person-hours.", y)
    ax.plot([LEFT, RIGHT], [y, y], color=BLACK, lw=.7)
    y -= .24
    cell_text(ax, LEFT + .04, y, "Loss pathway", size=8.4, bold=True)
    cell_text(ax, 2.75, y, "Ordinary square", size=8.2, bold=True)
    cell_text(ax, 5.02, y, "Dense commercial / arterial", size=8.0, bold=True)
    y -= .18
    ax.plot([LEFT, RIGHT], [y, y], color=RULE, lw=.6)
    loss_rows = (
        ("Cleaning / repair", "40 prem x THB5k/prem = THB0.20m",
         "150 prem x THB40k/prem = THB6.00m"),
        ("Business interruption", "20 shop-h x THB2k/shop-h = THB0.04m",
         "500 shop-h x THB8k/shop-h = THB4.00m"),
        ("Through-traveler delay", "400 p-h x THB100/p-h = THB0.04m",
         "10,000 p-h x THB150/p-h = THB1.50m"),
        ("Extra public response", "THB0.02m", "THB0.50m"),
        ("Illustrative total", "THB 0.30m", "THB 12.00m"),
    )
    for label, ordinary, dense in loss_rows:
        y -= .30
        bold = label == "Illustrative total"
        cell_text(ax, LEFT + .04, y, label, size=8.1, bold=bold)
        cell_text(ax, 2.75, y, ordinary, size=7.0, bold=bold)
        cell_text(ax, 5.02, y, dense, size=7.0, bold=bold)
        ax.plot([LEFT, RIGHT], [y-.14, y-.14], color=LIGHT, lw=.5)
    y -= .34
    y = para(ax, "The dense case assumes 150 affected premises, 100 businesses "
             "losing five hours of value added, and 10,000 additional hours "
             "lost by through-travelers. All unit values are assumptions; "
             "business and traveler hours must not overlap.", y,
             size=9.2, lead=.20)
    y = subsection(ax, "6.2", "Which sites enter the unfamiliar-cell shortlist?", y)
    y = para(ax, f"At the top 5% of unfamiliar cells, among 393 Q4 new-site "
             f"events in those cells, report "
             f"history selects {TOP5['history_hits']:.2f} and misses "
             f"{393-TOP5['history_hits']:.2f}. The proposed model selects "
             f"{TOP5['cnn_hits']:.2f} and misses "
             f"{393-TOP5['cnn_hits']:.2f}. At equal shortlist size it gains "
             f"{MODEL_ONLY:.2f} sites history missed, while dropping "
             f"{HISTORY_ONLY:.2f} that history selected: a net "
             f"{NET_EXTRA:+.2f} report-site events [11].", y)
    chart_top = y - .11
    ax.text(LEFT, chart_top, "RETROSPECTIVE SITE-SELECTION ACCOUNTING",
            size=7.6, family="DejaVu Sans", color=ACCENT,
            weight="bold", va="top")
    boxes = ((LEFT, "History selected", f"{TOP5['history_hits']:.2f}", "#83929D"),
             (2.48, "Added by model", f"+{MODEL_ONLY:.2f}", ACCENT),
             (4.19, "Displaced", f"-{HISTORY_ONLY:.2f}", WARM),
             (5.90, "Model selected", f"{TOP5['cnn_hits']:.2f}", BLUE))
    box_top = chart_top - .26
    for x, label, value, color in boxes:
        ax.add_patch(Rectangle((x, box_top - .85), 1.55, .85,
                               facecolor="#F5F8F8", edgecolor=RULE, lw=.7))
        ax.text(x + .775, box_top - .22, label, size=7.6, color=MID,
                family="DejaVu Sans", ha="center", va="center")
        ax.text(x + .775, box_top - .57, value, size=12.2, color=color,
                weight="bold", ha="center", va="center")
    for start in (2.33, 4.04, 5.75):
        arrow(ax, (start, box_top-.43), (start+.12, box_top-.43))
    y = fig_caption(ax, 6, "The model recovers some history-missed reported "
                    "sites and gives up others. Counts split cutoff ties "
                    "fractionally; they are not verified floods or prevented "
                    "losses.", box_top - 1.01)
    y = para(ax, "The economic question concerns the value of the added sites "
             "minus the value of displaced sites. It cannot assign either "
             "Table 1 loss scenario to all 366.20 sites missed by history: "
             "the model still misses most of them, and none has a measured "
             "avoidable loss.", y)
    finish(pdf, fig, y)

    # Page 7 — illustrative direct exposure and connected-cell spillovers.
    fig, ax = new_page(pdf, 7, "Economic interpretation")
    y = H - .93
    y = section(ax, "7", "Economic value at stake in overlooked areas", y)
    y = para(ax, f"The top-5% unfamiliar-cell screen adds {MODEL_ONLY:.2f} "
             f"later reported site-event equivalents and displaces "
             f"{HISTORY_ONLY:.2f} history picks. At THB 0.30m for every "
             "differently selected event, their illustrative exposure "
             f"difference is THB {BASE_EXPOSURE_DIFFERENCE_M:.2f}m. "
             "Bangkok's dense commercial and "
             "transport areas can be worth far more per affected square. "
             "This arithmetic assumes each site-event is a distinct flooded "
             "square; the reports do not establish that.", y)
    y = para(ax, f"A 2025 OpenStreetMap footprint overlay flags "
             f"{DENSE_MODEL_ONLY:.2f} added and {DENSE_HISTORY_ONLY:.2f} "
             "displaced site-event equivalents in cells with at least 20% "
             "mapped ground coverage [17]. Assign Table 1's THB 12m dense "
             "scenario to those flags and THB 0.30m elsewhere:", y)
    cards_top = y - .08
    value_cards = (
        (LEFT, "Added picks", f"THB {ADDED_EXPOSURE_M:.2f}m", ACCENT),
        (3.04, "Displaced picks", f"THB {DISPLACED_EXPOSURE_M:.2f}m", WARM),
        (5.32, "Difference", f"THB {EXPOSURE_DIFFERENCE_M:.2f}m", BLUE),
    )
    for x, label, value, color in value_cards:
        ax.add_patch(Rectangle((x, cards_top-.83), 2.19, .83,
                               facecolor="#EDF3F4", edgecolor=RULE, lw=.7))
        ax.text(x+1.095, cards_top-.23, label, size=8.0, color=MID,
                family="DejaVu Sans", ha="center", va="center")
        ax.text(x+1.095, cards_top-.58, value, size=12.0, color=color,
                weight="bold", ha="center", va="center")
    y = fig_caption(ax, 7, "Conditional discovery-pool sensitivity, not measured "
                    "damage or savings. Footprint coverage is a rough proxy: "
                    "it omits storeys, tenants, road traffic, and unmapped "
                    "buildings. Counts split shortlist ties fractionally.",
                    cards_top-.96)
    y = subsection(ax, "7.1", "When connected floods affect the wider city", y)
    y = para(ax, "Loss need not rise in proportion to flooded area. Several "
             "connected squares can close a through-route or delay access "
             "outside the wet footprint. Figure 8 starts with Table 1's "
             "THB 12m dense-square scenario and adds wider-network delay "
             "after a route is disrupted. It illustrates a threshold effect, "
             "not a measured Bangkok damage curve.", y)
    chart_top = y - .05
    chart_base = chart_top - 1.70
    plot_height = 1.25
    ax.text(LEFT, chart_top, "ILLUSTRATIVE LOSS, THB MILLION", size=7.7,
            family="DejaVu Sans", weight="bold", color=ACCENT, va="top")
    for level in (0, 50, 100, 150):
        yy = chart_base + level / 150 * plot_height
        ax.plot([LEFT+.38, RIGHT-.14], [yy, yy], color=LIGHT, lw=.65)
        ax.text(LEFT+.25, yy, str(level), size=7.2, color=MID,
                ha="right", va="center")
    for x, label, local, wider in (
        (1.78, "1 dense square", 12., 0),
        (4.05, "3 connected", 36., 3.0),
        (6.32, "10 connected", 120., 30.0),
    ):
        bar_width = .63
        ax.add_patch(Rectangle((x-bar_width/2, chart_base), bar_width,
                               local/150*plot_height, facecolor="#A9CBCD",
                               edgecolor="none"))
        if wider:
            ax.add_patch(Rectangle((x-bar_width/2,
                                    chart_base+local/150*plot_height),
                                   bar_width, wider/150*plot_height,
                                   facecolor=ACCENT, edgecolor="none"))
        ax.text(x, chart_base+(local+wider)/150*plot_height+.09,
                f"{local+wider:.1f}m", size=8.5, color=BLUE,
                weight="bold", ha="center", va="bottom")
        ax.text(x, chart_base-.12, label, size=8.0, color=BLACK,
                ha="center", va="top")
    ax.add_patch(Rectangle((4.64, chart_top-.10), .16, .10,
                           facecolor="#A9CBCD", edgecolor="none"))
    ax.text(4.86, chart_top-.05, "Local", size=7.1, color=MID, va="center")
    ax.add_patch(Rectangle((5.50, chart_top-.10), .16, .10,
                           facecolor=ACCENT, edgecolor="none"))
    ax.text(5.72, chart_top-.05, "Wider delay", size=7.1,
            color=MID, va="center")
    y = fig_caption(ax, 8, "Worked dense-corridor scenario. Three squares "
                    "add 20,000 person-hours of wider delay (THB 3m at THB "
                    "150/h); ten add 200,000 person-hours (THB 30m). These "
                    "assumptions go beyond the local delays in Table 1.",
                    chart_base-.27)
    y = para(ax, "The 45.85 added report sites are not known to form connected "
             "floods. Estimating real value requires verified extent, depth, "
             "duration, affected assets and travel, plus records of which "
             "early actions changed later outcomes and at what cost.", y)
    finish(pdf, fig, y)

    # Page 8 — data and simulation.
    fig, ax = new_page(pdf, 8, "Data priorities and simulation")
    y = H - .93
    y = section(ax, "8", "Decision value for district teams", y)
    y = para(ax, "The transparent report-history queue remains the reference. "
             "The proposed model ranks unfamiliar cells better on the report-site "
             "proxy, but at the top 5% of a full district queue both methods "
             "capture 1.27% of emerging sites [15]. A district should freeze "
             "full queues before storms and compare verified findings per "
             "staff-hour, travel, and actions in a new rainy season.", y)
    y = section(ax, "9", "From triage to rainfall-conditioned planning", y)
    y = para(ax, "The present site-ranking CNN contains no rain channel. "
             "A follow-up experiment added five rainfall windows strictly before "
             "each decision time, using cached hourly district-point estimates "
             "[14]. That earlier fixed-allocation test showed mixed gains "
             "across one-hour and six-hour decisions; its capture values are "
             "not comparable with the quota-free curve in Section 5. The "
             "evidence does not establish a dependable rain-intensity response. "
             "Neither ranker can predict flood extent under twice the rain "
             "without a new outcome and validation design.", y)
    y = table_caption(ax, 2, "Priority observations needed for a physical "
                      "flood-extent and preparedness model.", y)
    ax.plot([LEFT, RIGHT], [y, y], color=BLACK, lw=.8)
    y -= .22
    cell_text(ax, LEFT + .03, y, "Data class", size=8.3, bold=True)
    cell_text(ax, 2.45, y, "Required observation", size=8.3, bold=True)
    y -= .18
    ax.plot([LEFT, RIGHT], [y, y], color=RULE, lw=.6)
    reqs = [
        ("Storm forcing", "Hourly gauge/radar rain, timing, forecast cutoff, antecedent wetness"),
        ("Surface", "Detailed terrain, barriers, land cover, road and inlet position"),
        ("Drainage topology", "Verified inlet-pipe-canal-outfall connections and capacity"),
        ("Operating state", "Dated pump/gate logs, discharge, canal/river/sea levels"),
        ("Observed outcome", "Timed urban wet and dry cells, depth and valid observation footprint"),
    ]
    for name, desc in reqs:
        y -= .34
        cell_text(ax, LEFT + .03, y, name, size=8.0)
        cell_text(ax, 2.45, y, desc, size=7.9)
        ax.plot([LEFT, RIGHT], [y - .17, y - .17], color=LIGHT, lw=.5)
    y -= .35
    y = subsection(ax, "9.1", "Mapped drainage assets are a starting point", y)
    y = para(ax, "A public BMA ArcGIS service supplies 81,574 geocoded storm-drain "
             "pipe polylines, as well as canal, pump, and gate layers [12]. The "
             "geometry does not verify every street inlet, pipe junction, "
             "outfall, or event-time pump/gate state. Some invert elevations "
             "suggest candidate gravity direction, but their datum and line "
             "ordering require validation. A static pipe map alone cannot "
             "explain whether a particular storm can drain on time.", y)
    y = subsection(ax, "9.2", "A staged validation program", y)
    y = para(ax, "First, freeze the triage queue and measure later reported sites "
             "and field findings. Second, assemble strict time-matched rainfall, "
             "drainage state, and independent wet and dry observations; compare a "
             "rain-conditioned spatial model with simple land-cover/history "
             "rules and a hydraulic simulator on held-out storms and districts. "
             "Third, test scenario response to larger and longer storms with "
             "uncertainty ranges. Sentinel-1 radar can help with broad water "
             "extent under cloud, but narrow urban streets and valid dry "
             "observations need other sources [13].", y)
    finish(pdf, fig, y)

    # Page 9 — hackathon proposal, validation, conclusion.
    fig, ax = new_page(pdf, 9, "Prototype and BMA collaboration")
    y = H - .93
    y = section(ax, "10", "Hackathon prototype and BMA collaboration", y)
    y = subsection(ax, "10.1", "A map that makes the decision visible", y)
    y = para(ax, "The proposed hackathon demonstration lets a district choose "
             "a date and a fraction of unfamiliar cells to review, then "
             "compares two rankings: nearby-report history and the CNN. Reviewers can "
             "inspect each selected 250 m cell's report history, terrain, mapped "
             "roads, canals, and cover, and replay which later reported sites "
             "would have appeared within the same candidate fraction. The model, labels, "
             "and backtest are implemented; this interface is a proposed "
             "presentation layer, not a live flood alert.", y)
    y = subsection(ax, "10.2", "What to test together", y)
    y = para(ax, "A useful BMA collaboration would begin in shadow mode in one "
             "or more districts. Before storms, freeze BMA's existing risk-point "
             "or inspection queue and the model queue. After field visits, log "
             "verified conditions, depth where measurable, likely cause, crew "
             "time, action taken, and time to resolution. The primary comparison "
             "is newly verified problem sites per fixed staff-hour budget; "
             "secondary measures include total verified sites, travel, and "
             "response time. BMA's current FY2569 risk-point targets provide "
             "the operational context, not proof of model impact [3].", y)
    y = subsection(ax, "10.3", "Data partnership for the next capability", y)
    y = para(ax, "District feedback on false leads and overlooked sites would "
             "improve the triage model. Time-stamped radar or gauge rain, "
             "pump/gate operation, canal levels, and independently verified wet "
             "and dry locations would support a separate flood-extent model. "
             "That model would need held-out storms and districts, plus "
             "comparison with simple rules and a hydraulic simulator, before "
             "it could support larger-rain scenario planning.", y)
    top = y - .12
    for x, label in [(LEFT, "Hackathon replay\nranked cells + evidence"),
                     (3.11, "Joint shadow pilot\nverified sites + staff time"),
                     (5.46, "Scenario research\nrain + drainage + wet/dry")]:
        ax.add_patch(Rectangle((x, top - .72), 2.05, .72,
                               facecolor="#EDF3F4", edgecolor=RULE, lw=.7))
        ax.text(x + 1.025, top - .36, label, size=8.2, color=BLUE,
                ha="center", va="center", linespacing=1.5)
    arrow(ax, (2.84, top - .36), (3.07, top - .36))
    arrow(ax, (5.19, top - .36), (5.42, top - .36))
    y = fig_caption(ax, 9, "Proposed collaboration path. Each stage needs its own "
                    "measured outcome before the next use is justified.", top - .86)
    y = section(ax, "11", "Conclusion and evidence boundary", y)
    y = para(ax, "The promising result is narrower than a flood forecast: "
             "among unfamiliar cells, spatial context raised mean rank of "
             "later reported new sites by 5.53 percentile points in held-out "
             "districts. A no-reserve full queue showed no gain at the top 5% "
             "of all cells, so the useful field allocation remains unresolved. "
             "Q4 and the 250 m definition were examined during model development; "
             "unreported cells are unknown, not proven dry. These limits make "
             "a prospective, field-verified BMA pilot the decisive next step. "
             "The hackathon offers a working ranking method and a concrete "
             "way to evaluate public benefit together.", y)
    finish(pdf, fig, y)

    # Page 10 — primary and project references.
    fig, ax = new_page(pdf, 10, "References and source audit")
    y = H - .93
    ax.text(LEFT, y, "References and reproducibility", size=11.0,
            color=BLUE, weight="bold", va="top")
    y -= .34
    refs = [
        "[1] BMA Department of Drainage and Sewerage (DDS), Major-road flood statistics, 2025 CSV, incident 581 (13 Nov 2025). https://data.bangkok.go.th/dataset/frd_dds (accessed 24 Oct 2026).",
        "[2] BMA, FY2569 budget ordinance, DDS volume 69015, pp. 1 and 3. https://data.bangkok.go.th/dataset/2569 (accessed 24 Oct 2026).",
        "[3] BMA DDS, FY2569 annual action plan, pp. 2, 11-14 and 83. https://dds.bangkok.go.th/content/doc_download_count.php?id=11062 (accessed 24 Oct 2026).",
        "[4] FLOOD_CITYWIDE_HISTORY_RANKER.md and candidate_cells_by_district_250m.csv. Report-history benchmark and 10% shortlist-size audit.",
        "[5] NASA Shuttle Radar Topography Mission (SRTM), elevation product; feature audit in FLOOD_CURRENT_STATE.md.",
        "[6] OpenStreetMap contributors, road and waterway geometry; feature audit in FLOOD_CURRENT_STATE.md.",
        "[7] ESA WorldCover 2021, land-cover product; feature audit in FLOOD_CURRENT_STATE.md.",
        "[8] FLOOD_CITYWIDE_PHYSICAL_SITE_TEST.md and q4_2025_physical_site_final.json. Site labels, model arms, fold selection, intervals.",
        "[9] FLOOD_CITYWIDE_SPATIAL_NOVEL_QUEUE.md and flood_citywide_spatial_novel_queue.py. Early CNN and nearby score.",
        "[10] FLOOD_CITYWIDE_SPATIAL_MASK_AUGMENTATION.md. Earlier center and block-mask experiments.",
        "[11] FLOOD_SITE_QUOTA_FREE_RANKING.md, q4_2025_site_novel_rank_curve.json, and q4_2025_site_novel_rank_curve_events.csv. Quota-free source selection, held-out ranking, site-level shortlist switches, and subgroup counts.",
        "[12] FLOOD_BANGKOK_DRAINAGE_GIS.md. Audit of the BMA ArcGIS Thematic/Drainage service, including pipe and canal layers.",
        "[13] FLOOD_CURRENT_STATE.md and FLOOD_OPEN_DATA_AUDIT_2026.md. Wider model and observation limits.",
        "[14] FLOOD_CITYWIDE_SITE_RAIN_HOURLY.md. Matched pre-cutoff hourly-rain CNN experiment and five-fold results.",
        "[15] FLOOD_SITE_ALL_CELL_CURVES.md and q4_2025_site_all_cell_curves.json. No-reserve full district-cell capture curves for all and emerging reported sites.",
        "[16] FLOOD_SITE_DISTANCE_CURVES.md and q4_2025_site_distance_rank_curves.json. Emerging-site capture within unfamiliar cells by distance to earlier reports.",
        "[17] osm_2025_building_footprint_citywide_250m.npz and accompanying audit JSON, derived by flood_citywide_surface_osm.py from the 1 Jan 2025 OpenStreetMap snapshot. Mapped ground-footprint sensitivity in Section 7.",
    ]
    for ref in refs:
        y = para(ax, ref, y, x=LEFT + .02, width=WIDTH - .05,
                 size=8.7, lead=.18, after=.08, color=BLACK)
    y -= .09
    ax.plot([LEFT, RIGHT], [y, y], color=RULE, lw=.7)
    y -= .22
    y = para(ax, "Reproducibility. The generator build_flood_site_academic_paper.py "
             "reads the committed rank-curve JSON files, site-event CSV, and OSM "
             "footprint NPZ to rebuild this PDF. The model "
             "fitting and evaluation scripts are named in references [8,11]. "
             "Re-running those analyses requires source data listed in the "
             "repository README; the committed results alone reproduce the paper.",
             y, size=8.3, lead=.17, color=MID)
    ax.text(LEFT + .02, y-.03, f"Repository: {REPO_URL}", size=8.6,
            color=ACCENT, va="top", url=REPO_URL)
    y -= .26
    finish(pdf, fig, y)

print(OUT)
