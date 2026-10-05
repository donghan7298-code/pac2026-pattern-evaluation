"""Deterministic geometry features; safety cannot be traded for score."""
import math
from collections import deque

import numpy as np

from pacdata.packing import apply_placement, bounds, dims, mask
from pacdata.teacher import FEATURE_NAMES as BASE_NAMES, features as base_features

EXTRA_NAMES = ["space_quality", "largest_free_patch", "free_fragmentation", "height_variation",
               "remaining_footprint_fit", "weakest_safety", "load_margin", "balance",
               "handling_cost", "buffer_occupancy", "buffer_previous_moves", "support_ratio"]
FEATURE_NAMES = list(BASE_NAMES) + EXTRA_NAMES
WEIGHT_NAMES = ["space", "future_fit", "future_volume", "risk", "balance", "handling"]


def normalized_weights(weights):
    if set(weights) != set(WEIGHT_NAMES):
        raise ValueError("weights must contain exactly " + ", ".join(WEIGHT_NAMES))
    values = {k: float(weights[k]) for k in WEIGHT_NAMES}
    if any(not math.isfinite(v) or v < 0 for v in values.values()) or sum(values.values()) <= 0:
        raise ValueError("weights must be finite, nonnegative and have a positive sum")
    total = sum(values.values())
    return {k: v / total for k, v in values.items()}


def height_grid(pallet, boxes, side=6):
    xs = (np.arange(side) + .5) * pallet["length_m"] / side
    ys = (np.arange(side) + .5) * pallet["width_m"] / side
    grid = np.zeros((side, side))
    for box in boxes:
        lo, hi = bounds(box)
        cover = ((ys[:, None] >= lo[1]) & (ys[:, None] <= hi[1]) &
                 (xs[None, :] >= lo[0]) & (xs[None, :] <= hi[0]))
        grid[cover] = np.maximum(grid[cover], hi[2])
    return grid


def free_patches(grid):
    """Grid proxy for connected floor space, not an exact collision test."""
    free = grid < 1e-8
    visited = set()
    sizes = []
    h, w = free.shape
    for y in range(h):
        for x in range(w):
            if not free[y, x] or (y, x) in visited:
                continue
            queue = deque([(y, x)])
            visited.add((y, x)); size = 0
            while queue:
                yy, xx = queue.popleft(); size += 1
                for yn, xn in [(yy-1, xx), (yy+1, xx), (yy, xx-1), (yy, xx+1)]:
                    if 0 <= yn < h and 0 <= xn < w and free[yn, xn] and (yn, xn) not in visited:
                        visited.add((yn, xn)); queue.append((yn, xn))
            sizes.append(size)
    free_count = sum(sizes)
    largest = max(sizes, default=0) / max(1, free_count)
    fragmentation = (len(sizes)-1) / max(1, free_count) if sizes else 0.
    return largest, fragmentation


def safety_metrics(pallet, boxes):
    total = sum(b["mass_kg"] for b in boxes)
    limits = [pallet["length_m"], pallet["width_m"], pallet["max_height_m"]]
    cog = [sum(b["mass_kg"] * (b["position_m"][a]+dims(b, b["yaw_deg"])[a]/2)
               for b in boxes) / total / limits[a] for a in range(3)] if boxes else [.5, .5, 0.]
    edge = min(2*cog[0], 2*(1-cog[0]), 2*cog[1], 2*(1-cog[1]))
    margins = [(b["top_load_capacity_N"] - b.get("upper_load_N", 0.)) /
               max(1e-9, b["top_load_capacity_N"]) for b in boxes if b.get("upper_load_N", 0.) > 0]
    load_margin = min(margins, default=1.)
    return dict(cog=cog, cog_edge=edge, load_margin=load_margin,
                balance=max(0., 1.-abs(cog[0]-.5)-abs(cog[1]-.5)))


def static_components(obs, candidate, config):
    p = obs["pallet"]; box = obs["current_box"]
    post = apply_placement(p, obs["placed_boxes"], box, candidate)
    safety = safety_metrics(p, post)
    limits = config["safety"]
    if safety["cog_edge"] + 1e-8 < limits["min_cog_edge"]:
        raise ValueError("COG_EDGE_MINIMUM")
    # Full support is already a hard constraint of the shared conservative core.
    weakest = min(1., safety["cog_edge"] / limits["target_cog_edge"],
                  max(0., safety["load_margin"]) / limits["target_load_margin"])
    grid = height_grid(p, post)
    largest, fragmentation = free_patches(grid)
    variation = min(1., float(np.std(grid)) / p["max_height_m"])
    # Axis-aligned empty floor rectangles; coarse proxy for the remaining SKU fit.
    fits = 0.; count = 0
    for sku, n in obs["unseen_inventory"].items():
        item = obs["catalog"][sku]; count += n; can_fit = False
        for yaw in item["allowed_yaw_deg"]:
            d = dims(item, yaw)
            cells_x = math.ceil(d[0] / p["length_m"] * grid.shape[1])
            cells_y = math.ceil(d[1] / p["width_m"] * grid.shape[0])
            for y in range(grid.shape[0]-cells_y+1):
                for x in range(grid.shape[1]-cells_x+1):
                    if not np.any(grid[y:y+cells_y, x:x+cells_x]):
                        can_fit = True; break
                if can_fit: break
            if can_fit: break
        fits += n * can_fit
    footprint_fit = fits / count if count else 1.
    space = max(0., min(1., .35*largest + .35*footprint_fit + .2*(1.-variation) +
                       .1*(1.-min(1., fragmentation*10))))
    xyz = candidate["position_m"]
    tp = config["time_proxy"]
    seconds = tp["base_seconds"] + tp["xy_seconds"]*(xyz[0]/p["length_m"]+xyz[1]/p["width_m"]) + tp["z_seconds"]*xyz[2]/p["max_height_m"]
    return dict(space=space, largest_free_patch=largest, fragmentation=fragmentation,
                height_variation=variation, footprint_fit=footprint_fit, safety=weakest,
                load_margin=safety["load_margin"], cog_edge=safety["cog_edge"],
                balance=safety["balance"], handling=min(1., seconds/20.),
                handling_seconds_proxy=seconds, support_ratio=1.)


def feature_vector(obs, candidate, static):
    # Keep the repository's normalized camera/inventory representation intact.
    v = base_features(obs, candidate)
    v += [static["space"], static["largest_free_patch"], static["fragmentation"],
          static["height_variation"], static["footprint_fit"], static["safety"],
          static["load_margin"], static["balance"], static["handling"],
          len(obs.get("buffer_boxes", []))/max(1, obs.get("buffer_capacity", 2)),
          float(obs["current_box"].get("buffer_moves", 0)), static["support_ratio"]]
    return v


def efficiency_score(static, future, weights):
    return (weights["space"]*static["space"] + weights["future_fit"]*future["mean_fit"] +
            weights["future_volume"]*future["mean_volume"] - weights["risk"]*future["risk"] +
            weights["balance"]*static["balance"] - weights["handling"]*static["handling"])


def priority(static, score):
    """Below target, weakest safety precedes efficiency; above target it saturates."""
    safe = static["safety"] >= 1.-1e-8
    return (int(safe), score if safe else static["safety"], score)


def valid_candidates(obs, config, supplied=None):
    from pacdata.packing import generate_candidates
    p, placed, box = obs["pallet"], obs["placed_boxes"], obs["current_box"]
    # No current-candidate cap before Teacher or Ranking. Finite EP search only.
    candidates = supplied if supplied is not None else generate_candidates(p, placed, box, 10**9)
    rows, rejected = [], []
    seen = set()
    for index, c in enumerate(candidates):
        c = dict(c, candidate_id=str(c.get("candidate_id", index)))
        key = (tuple(c.get("position_m", [])), c.get("yaw_deg"))
        if key in seen: continue
        seen.add(key)
        verdict = mask(p, placed, box, c)
        if verdict["status"] != "ALLOW":
            rejected.append(dict(candidate_id=c["candidate_id"], **verdict)); continue
        try:
            static = static_components(obs, c, config)
        except ValueError as error:
            rejected.append(dict(candidate_id=c["candidate_id"], status="REJECT", reasons=[str(error)])); continue
        rows.append(dict(candidate=c, static=static))
    return rows, rejected
