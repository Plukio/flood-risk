"""Conservative synthetic surface/pipe step for spatial pretraining.

Water continuity and passive hydraulic-head direction are hard constraints.
Infiltration, pipe entry, mobility and outlet rates are illustrative scenario
assumptions, not measured Bangkok operation or observed flood labels.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn
from torch.nn import functional as F

from flood_citywide_spatial_novel_queue import SpatialNovelCNN

CELL_AREA_M2 = 250.**2
FLOW_NAMES = ("north", "south", "west", "east")


def neighbor_fields(head: np.ndarray, candidate: np.ndarray):
    shape = head.shape
    neighbor = np.zeros((4, *shape), np.float32)
    valid = np.zeros((4, *shape), bool)
    neighbor[0, 1:] = head[:-1]
    valid[0, 1:] = candidate[:-1]
    neighbor[1, :-1] = head[1:]
    valid[1, :-1] = candidate[1:]
    neighbor[2, :, 1:] = head[:, :-1]
    valid[2, :, 1:] = candidate[:, :-1]
    neighbor[3, :, :-1] = head[:, 1:]
    valid[3, :, :-1] = candidate[:, 1:]
    return neighbor, valid


def shift_incoming_np(flows: np.ndarray):
    incoming = np.zeros(flows.shape[1:], np.float32)
    incoming[:-1] += flows[0, 1:]
    incoming[1:] += flows[1, :-1]
    incoming[:, :-1] += flows[2, :, 1:]
    incoming[:, 1:] += flows[3, :, :-1]
    return incoming


def scenario(rng, candidate, pipe_proximity):
    """Random initial water, rainfall pulse, pipe storage and open fraction."""
    shape = candidate.shape
    noise = gaussian_filter(rng.random(shape).astype(np.float32),
                            float(rng.uniform(3, 12)), mode="reflect")
    low, high = np.quantile(noise, [.3, .98])
    storm = np.clip((noise-low)/max(float(high-low), 1e-6), 0, 1)
    rain = (rng.uniform(.005, .05)*storm*candidate).astype(np.float32)
    wet = gaussian_filter(rng.random(shape).astype(np.float32), 4)
    wet = np.clip((wet-np.quantile(wet, .4))/max(float(np.quantile(wet,.98)-
                                                  np.quantile(wet,.4)), 1e-6), 0, 1)
    surface = (rng.uniform(0, .025)*wet*candidate).astype(np.float32)
    pipe = (rng.uniform(0, .025)*pipe_proximity*candidate).astype(np.float32)
    gate = np.full(shape, rng.choice([.2, .5, 1.]), np.float32)*candidate
    return surface, rain, pipe, gate


def teacher_step(surface, rain, pipe, gate, elevation, candidate, cover,
                 building, roads, drainage):
    """One mass-conserving heuristic finite-volume step; units are metres."""
    available = (surface+rain)*candidate
    built, vegetation = cover
    pipe_prox, outlet_access, pump_prox = drainage
    infiltration_fraction = np.clip(
        .005+.11*vegetation*(1-built)*(1-building), 0, .16)
    infiltrated = available*infiltration_fraction*candidate
    pipe_fraction = np.clip((.025+.16*pipe_prox+.025*pump_prox)*
                            (pipe_prox>.1), 0, .23)
    pipe_entry = (available-infiltrated)*pipe_fraction*candidate
    remaining = available-infiltrated-pipe_entry
    head = elevation+available
    neighbors, valid = neighbor_fields(head, candidate)
    downhill = np.sqrt(np.maximum(head[None]-neighbors, 0))*valid
    # Roads and mapped buildings affect uncertain effective mobility; they
    # are not asserted to be measured barriers or known runoff coefficients.
    mobility = np.clip(.20+.10*roads-.08*vegetation-.04*building, .08, .38)
    weights = downhill*(1+.12*roads[None])
    denominator = weights.sum(axis=0)
    fraction = np.divide(weights, denominator[None],
                         out=np.zeros_like(weights), where=denominator[None]>0)
    outgoing = remaining[None]*mobility[None]*fraction
    next_surface = remaining-outgoing.sum(axis=0)+shift_incoming_np(outgoing)
    pipe_available = pipe+pipe_entry
    outlet_rate = np.clip((.08+.35*outlet_access+.06*pump_prox)*
                          np.where(outlet_access>.01, outlet_access, 0)*gate,
                          0, .55)
    outlet = pipe_available*outlet_rate
    next_pipe = pipe_available-outlet
    result = {"surface": next_surface.astype(np.float32),
              "pipe": next_pipe.astype(np.float32),
              "infiltration": infiltrated.astype(np.float32),
              "outlet": outlet.astype(np.float32),
              "flow": outgoing.astype(np.float32),
              "available": available.astype(np.float32)}
    check = (float((available+pipe).sum()) -
             float((next_surface+next_pipe+infiltrated+outlet).sum()))
    if abs(check) > .005 or np.min(next_surface) < -1e-7 or np.min(next_pipe) < -1e-7:
        raise ValueError(f"Synthetic step violates continuity: {check}")
    return result


class ConservativeFlowCNN(SpatialNovelCNN):
    """Predict transfers; continuity and passive head direction hold exactly."""

    def __init__(self):
        super().__init__()
        # 0 after-rain surface water, 1 pipe water, 2 gate scenario;
        # 3:13 LiDAR/OSM/WorldCover, 13 candidate, 14:18 BMA GIS,
        # 18 OSM buildings, 19 valid mask.
        self.entry = nn.Conv2d(20, 8, 1)
        self.flow_head = nn.Conv2d(8, 8, 1)
        with torch.no_grad():
            self.flow_head.bias.zero_()
            self.flow_head.bias[0] = 2.5  # retain most surface water initially
            self.flow_head.bias[5:7] = -2.5  # modest initial infiltration/entry

    def features(self, image):
        valid = image[:, -1:]
        features = F.relu(self.entry(image))*valid
        for step in self.steps:
            update = step["mix"](F.relu(step["spatial"](features)))
            features = F.relu(features+.25*update)*valid
        return features

    def forward(self, image, available, pipe, elevation):
        logits = self.flow_head(self.features(image))
        candidate = image[:, 13:14] > .5
        head = elevation+available
        directions = []
        allowed = []
        north = torch.zeros_like(head)
        north[:, :, 1:] = head[:, :, :-1]
        directions.append(north)
        north_ok = torch.zeros_like(candidate)
        north_ok[:, :, 1:] = candidate[:, :, :-1]
        allowed.append(north_ok)
        south = torch.zeros_like(head)
        south[:, :, :-1] = head[:, :, 1:]
        directions.append(south)
        south_ok = torch.zeros_like(candidate)
        south_ok[:, :, :-1] = candidate[:, :, 1:]
        allowed.append(south_ok)
        west = torch.zeros_like(head)
        west[:, :, :, 1:] = head[:, :, :, :-1]
        directions.append(west)
        west_ok = torch.zeros_like(candidate)
        west_ok[:, :, :, 1:] = candidate[:, :, :, :-1]
        allowed.append(west_ok)
        east = torch.zeros_like(head)
        east[:, :, :, :-1] = head[:, :, :, 1:]
        directions.append(east)
        east_ok = torch.zeros_like(candidate)
        east_ok[:, :, :, :-1] = candidate[:, :, :, 1:]
        allowed.append(east_ok)
        downhill = torch.cat([(a & (head>z)) for a,z in zip(allowed,directions)],1)
        pipe_allowed = candidate & (image[:, 14:15]>.1)
        choices = torch.cat((logits[:, :1],
            logits[:, 1:5].masked_fill(~downhill, -1e5),
            logits[:, 5:6],
            logits[:, 6:7].masked_fill(~pipe_allowed, -1e5)),1)
        share = torch.softmax(choices, dim=1)
        transfer = available*share
        outgoing = transfer[:, 1:5]
        incoming = (F.pad(outgoing[:, 0:1, 1:], (0,0,0,1))+
                    F.pad(outgoing[:, 1:2, :-1], (0,0,1,0))+
                    F.pad(outgoing[:, 2:3, :, 1:], (0,1,0,0))+
                    F.pad(outgoing[:, 3:4, :, :-1], (1,0,0,0)))
        next_surface = transfer[:, :1]+incoming
        infiltrated = transfer[:, 5:6]
        pipe_entry = transfer[:, 6:7]
        pipe_available = pipe+pipe_entry
        outlet_access = torch.where(image[:, 15:16]>.01,
                                    image[:, 15:16], 0.)
        gate = image[:, 2:3].clamp(0,1)
        outlet = pipe_available*torch.sigmoid(logits[:, 7:8])*outlet_access*gate
        next_pipe = pipe_available-outlet
        return {"surface": next_surface, "pipe": next_pipe,
                "infiltration": infiltrated, "outlet": outlet,
                "flow": outgoing}


def physics_self_check():
    torch.manual_seed(3)
    model = ConservativeFlowCNN().eval()
    image = torch.zeros((1, 20, 7, 7))
    image[:, 13] = 1
    image[:, 14] = 1
    image[:, 15] = 1
    image[:, 19] = 1
    image[:, 2] = 1
    available = torch.zeros((1, 1, 7, 7))
    available[:, :, 3, 3] = .04
    pipe = torch.zeros_like(available)
    elevation = torch.ones_like(available)
    elevation[:, :, 3, 3] = 1.2
    elevation[:, :, 3, 4] = 1.0
    with torch.no_grad():
        result = model(image, available, pipe, elevation)
    total = sum(float(result[key].sum()) for key in
                ("surface", "pipe", "infiltration", "outlet"))
    if abs(total-.04) > 1e-6 or torch.min(result["surface"]) < 0:
        raise ValueError("Conservative CNN lost or created water")
    # The eastern cell is lower; the other three neighboring heads are lower
    # too in this toy setup. Raising one neighbor above source forbids its flow.
    elevation[:, :, 2, 3] = 1.4
    with torch.no_grad():
        blocked = model(image, available, pipe, elevation)
    if float(blocked["flow"][0, 0, 3, 3]) != 0:
        raise ValueError("Passive model moved water to higher hydraulic head")
