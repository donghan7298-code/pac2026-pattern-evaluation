"""Shared scenario samples for all candidates; no oracle arrival order."""
import copy
import math
import random

import numpy as np

from pacdata.io import stable_seed
from pacdata.packing import apply_placement, generate_candidates, immediate_score
from .scoring import safety_metrics

SCENARIOS = ["random", "large_late", "heavy_late", "small_early", "sku_cluster", "fragmentation"]
OUTPUT_NAMES = ["mean_fit", "mean_volume", "cvar_fit", "block_rate"]


def sample_futures(obs, config):
    depth = config["future_depth"]; result = []
    pool = [sku for sku, count in obs["unseen_inventory"].items() for _ in range(count)]
    for draw in range(config["future_draws"]):
        rng = random.Random(stable_seed(config["seed"], obs.get("request_id", "request"), draw))
        prefix = []
        for t in obs["observed_preview"][:depth]:
            d = [max(.01, t["dimensions_m"][a]+rng.gauss(0, t.get("dimension_sigma_m", 0.))) for a in (0, 1)]
            prefix.append(dict(box_id=t["track_id"], sku=t["sku"], dimensions_m=d+[t["dimensions_m"][2]],
                mass_kg=t["nominal_mass_kg"], top_load_capacity_N=t["top_load_capacity_N"],
                allowed_yaw_deg=t["allowed_yaw_deg"], measurement_valid=True, visual_damage_observed=False))
        unseen = list(pool); rng.shuffle(unseen)
        scenario = SCENARIOS[draw % len(SCENARIOS)]
        cat = obs["catalog"]
        if scenario in ("large_late", "small_early"):
            unseen.sort(key=lambda sku: math.prod(cat[sku]["dimensions_m"]))
        elif scenario == "heavy_late": unseen.sort(key=lambda sku: cat[sku]["mass_kg"])
        elif scenario == "sku_cluster": unseen.sort()
        elif scenario == "fragmentation":
            ordered = sorted(unseen, key=lambda sku: math.prod(cat[sku]["dimensions_m"])); unseen = []
            while ordered:
                unseen.append(ordered.pop(0))
                if ordered: unseen.append(ordered.pop())
        # A late large/heavy box must occur inside the bounded horizon as a stress probe.
        n = max(0, depth-len(prefix))
        if scenario in ("large_late", "heavy_late") and len(unseen) > n > 1:
            unseen = unseen[:n-1]+unseen[-1:]
        for i, sku in enumerate(unseen[:n]):
            b = copy.deepcopy(cat[sku])
            b.update(box_id=f"future_{draw}_{i}", measurement_valid=True, visual_damage_observed=False)
            b["dimensions_m"] = [v*rng.uniform(.99, 1.01) for v in b["dimensions_m"]]
            prefix.append(b)
        result.append(dict(scenario=scenario, boxes=prefix))
    return result


def future_metrics(obs, candidate, futures, config):
    pallet = obs["pallet"]
    post = apply_placement(pallet, obs["placed_boxes"], obs["current_box"], candidate)
    fits, volumes, blocks = [], [], []
    pv = pallet["length_m"]*pallet["width_m"]*pallet["max_height_m"]
    for scenario in futures:
        sequence = scenario["boxes"]; state = post; count = 0; volume = 0.
        for box in sequence:
            candidates = generate_candidates(pallet, state, box, config["future_candidate_limit"])
            candidates.sort(key=lambda c: immediate_score(pallet, state, box, c), reverse=True)
            next_state = None
            for c in candidates:
                proposal = apply_placement(pallet, state, box, c)
                if safety_metrics(pallet, proposal)["cog_edge"]+1e-8 >= config["safety"]["min_cog_edge"]:
                    next_state = proposal; break
            if next_state is None: break
            state = next_state
            count += 1; volume += math.prod(box["dimensions_m"])
        fits.append(count/len(sequence) if sequence else 1.)
        volumes.append(volume/pv); blocks.append(float(count < len(sequence)))
    mean = float(np.mean(fits)); tail_n = max(1, math.ceil(len(fits)*config["cvar_fraction"]))
    cvar = float(np.mean(sorted(fits)[:tail_n])); block = float(np.mean(blocks))
    return dict(mean_fit=mean, mean_volume=float(np.mean(volumes)), cvar_fit=cvar,
                block_rate=block, worst_fit=min(fits), risk=min(1., mean-cvar+.5*block),
                scenario_fit=fits)
