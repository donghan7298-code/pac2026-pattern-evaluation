"""Reproducible group splits, full-candidate labels and paired closed-loop evaluation."""
import copy
import math
import time

import numpy as np

from pacdata.generate import CAMERAS, MODES, episodes_for_group
from pacdata.observe import observe
from pacdata.packing import apply_placement, mask

from .planner import Planner
from .rollout import future_metrics, sample_futures
from .scoring import efficiency_score, feature_vector, priority, safety_metrics, valid_candidates


def make_episodes(indices, split, seed, modes=None, cameras=None, boxes=18):
    config = dict(seed=seed, min_boxes=boxes, max_boxes=boxes+4)
    result = []
    for index in indices:
        for episode in episodes_for_group(index, config, split):
            if modes is not None and episode["scenario"] not in modes: continue
            if cameras is not None and episode["camera"]["name"] not in cameras: continue
            result.append(episode)
    return result


def collect_labels(episodes, config, max_states=5):
    groups = []; started = time.perf_counter()
    for ei, episode in enumerate(episodes):
        state = dict(processed=0, pallet=copy.deepcopy(episode["pallet"]), placed=[], remeasured=[], routed=0)
        collected = 0
        for step in range(len(episode["oracle"]["arrivals"])):
            obs = observe(episode, state); box = obs["current_box"]
            if not box["measurement_valid"]:
                state["remeasured"].append(step); obs = observe(episode, state); box = obs["current_box"]
            if box["visual_damage_observed"]:
                state["processed"] += 1; continue
            rows, _ = valid_candidates(obs, config)
            if not rows: break
            if collected < max_states and step % 2 == 0:
                futures = sample_futures(obs, config)
                for row in rows:
                    row["features"] = feature_vector(obs, row["candidate"], row["static"])
                    row["future"] = future_metrics(obs, row["candidate"], futures, config)
                group = dict(base_group=episode["base_group"], episode_id=episode["episode_id"],
                    split=episode["split"], scenario=episode["scenario"], request_id=obs["request_id"],
                    generated_valid_candidates=len(rows), labeled_candidates=len(rows), rows=rows)
                groups.append(group); collected += 1
                winner = max(rows, key=lambda r: priority(r["static"], efficiency_score(r["static"], r["future"], config["weights"])))
            else:
                # Include low-quality visited states so ranking is not trained only on its own Teacher.
                winner = max(rows, key=lambda r: priority(r["static"], r["static"]["space"]))
            state["placed"] = apply_placement(state["pallet"], state["placed"], box, winner["candidate"])
            state["processed"] += 1
            for event in episode["oracle"]["events"]:
                if event["at_processed"] == state["processed"]: state["pallet"] = copy.deepcopy(event["new_pallet"])
        if (ei+1) % 8 == 0 or ei+1 == len(episodes):
            print(f"Teacher {episode['split']}: {ei+1}/{len(episodes)} episodes, {len(groups)} states, {time.perf_counter()-started:.1f}s", flush=True)
    return groups


def run_episode(episode, planner, mode="ahead", buffer=False, keep_trace=False):
    state = dict(processed=0, pallet=copy.deepcopy(episode["pallet"]), placed=[], remeasured=[], routed=0)
    buffered = []; decisions = []; trace = []; extra_seconds = 0.; buffer_moves = 0; handling = 0.
    violations = 0; minimum_safety = 1.; close_reason = "COMPLETE"
    total_arrivals = len(episode["oracle"]["arrivals"])
    for _ in range(total_arrivals*3+10):
        obs = observe(episode, state)
        obs.update(buffer_boxes=copy.deepcopy(buffered), buffer_capacity=planner.config["buffer"]["capacity"],
                   stream_terminated=state["processed"] == total_arrivals)
        result = planner.plan(obs, mode=mode, allow_buffer=buffer); action = result["action"]
        kind = action["type"]; decisions.append(result["decision_ms"])
        if keep_trace: trace.append(dict(step=state["processed"], **result))
        if kind == "REMEASURE": state["remeasured"].append(state["processed"]); extra_seconds += 2.; continue
        if kind == "ROUTE_NG": state["routed"] += 1; state["processed"] += 1
        elif kind == "BUFFER_CURRENT":
            buffered.append(dict(obs["current_box"], buffer_moves=1))
            buffer_moves += 1; state["processed"] += 1
            extra_seconds += planner.config["buffer"]["move_seconds"]
        elif kind in ("PLACE_CURRENT", "RETRIEVE_BUFFER"):
            box = obs["current_box"] if kind == "PLACE_CURRENT" else next(b for b in buffered if b["box_id"] == action["box_id"])
            if mask(state["pallet"], state["placed"], box, action["candidate"])["status"] != "ALLOW": violations += 1
            state["placed"] = apply_placement(state["pallet"], state["placed"], box, action["candidate"])
            minimum_safety = min(minimum_safety, result["static"]["safety"])
            handling += result["static"]["handling_seconds_proxy"]
            if kind == "RETRIEVE_BUFFER":
                buffered = [b for b in buffered if b["box_id"] != box["box_id"]]
                extra_seconds += planner.config["buffer"]["move_seconds"]
            else: state["processed"] += 1
        elif kind == "DONE": break
        elif kind in ("PALLET_CLOSE", "HOLD", "WAIT"):
            close_reason = action.get("reason", kind); break
        else: raise ValueError("Unknown engine action: " + kind)
        extra_seconds += len(buffered)*planner.config["buffer"]["occupancy_seconds"]
        for event in episode["oracle"]["events"]:
            if event["at_processed"] == state["processed"]: state["pallet"] = copy.deepcopy(event["new_pallet"])
    pallet = state["pallet"]
    # Independently rebuild load propagation and masks from the placement history.
    rebuilt = []
    for box in state["placed"]:
        candidate = dict(position_m=box["position_m"], yaw_deg=box["yaw_deg"])
        try:
            rebuilt = apply_placement(pallet, rebuilt, box, candidate)
        except ValueError:
            violations += 1; break
    if len(rebuilt) == len(state["placed"]):
        violations += sum(abs(a["upper_load_N"]-b["upper_load_N"]) > 1e-6 for a, b in zip(rebuilt, state["placed"]))
    volume = sum(math.prod(b["dimensions_m"]) for b in state["placed"])
    eligible = sum(not b["damaged"] for b in episode["oracle"]["arrivals"])
    safety = safety_metrics(pallet, state["placed"])
    row = dict(episode_id=episode["episode_id"], base_group=episode["base_group"], split=episode["split"],
        scenario=episode["scenario"], camera=episode["camera"]["name"], policy=mode+("+buffer" if buffer else ""),
        placed=len(state["placed"]), arrivals=total_arrivals, routed=state["routed"],
        volume_ratio=volume/(pallet["length_m"]*pallet["width_m"]*pallet["max_height_m"]),
        success_rate=len(state["placed"])/max(1, eligible), blocked=len(state["placed"])+state["routed"] < total_arrivals,
        mask_violations=violations, minimum_safety=minimum_safety, cog_edge=safety["cog_edge"],
        load_margin=safety["load_margin"], buffer_moves=buffer_moves, leftover_buffer=len(buffered),
        handling_seconds_proxy=handling+extra_seconds, decision_ms=decisions, close_reason=close_reason)
    if keep_trace: row["trace"] = trace
    return row


def summarize(rows):
    ms = [v for r in rows for v in r["decision_ms"]]
    group_values = {}
    for row in rows: group_values.setdefault(row["base_group"], []).append(row["volume_ratio"])
    return dict(episodes=len(rows), base_groups=len(group_values),
        mean_volume_ratio=float(np.mean([r["volume_ratio"] for r in rows])),
        mean_success_rate=float(np.mean([r["success_rate"] for r in rows])),
        worst_volume_ratio=min(r["volume_ratio"] for r in rows),
        p10_volume_ratio=float(np.quantile([r["volume_ratio"] for r in rows], .1)),
        worst_group_mean_volume_ratio=min(float(np.mean(v)) for v in group_values.values()),
        blocked_episode_rate=float(np.mean([r["blocked"] for r in rows])),
        mask_violations=sum(r["mask_violations"] for r in rows),
        min_load_margin=min(r["load_margin"] for r in rows),
        min_safety=min(r["minimum_safety"] for r in rows),
        mean_decision_ms=float(np.mean(ms)), p95_decision_ms=float(np.quantile(ms, .95)),
        mean_buffer_moves=float(np.mean([r["buffer_moves"] for r in rows])),
        mean_handling_seconds_proxy=float(np.mean([r["handling_seconds_proxy"] for r in rows])))


def validation_objective(summary):
    if summary["mask_violations"] or summary["min_load_margin"] < -1e-8: return -1e9
    # External completed-episode metrics, not the policy's own candidate scores.
    return summary["mean_volume_ratio"]+.2*summary["mean_success_rate"]+.1*summary["p10_volume_ratio"]


def tune_weights(episodes, model, config, rounds=2, trials_per_round=10):
    from .scoring import WEIGHT_NAMES, normalized_weights
    rng = np.random.default_rng(config["seed"]+1); records = []
    planner = Planner(config, model, use_model=False)
    planner.cache_limit = 10000
    center = np.asarray([config["weights"][k] for k in WEIGHT_NAMES])
    candidates = [center, np.asarray([.12,.6,.15,.07,.03,.03]), np.asarray([.25,.3,.2,.2,.03,.02])]
    for round_index in range(rounds):
        candidates += list(rng.dirichlet(np.maximum(center*12, .5), size=trials_per_round))
        for vector in candidates:
            weights = normalized_weights(dict(zip(WEIGHT_NAMES, map(float, vector))))
            planner.config["weights"] = weights
            rows = [run_episode(ep, planner) for ep in episodes]; summary = summarize(rows)
            record = dict(round=round_index+1, weights=weights, metrics=summary, objective=validation_objective(summary))
            records.append(record)
            print(f"weight search {len(records)}: objective={record['objective']:.5f}, fill={summary['mean_volume_ratio']:.4f}", flush=True)
        elites = sorted(records, key=lambda r: r["objective"], reverse=True)[:max(3, len(records)//5)]
        center = np.mean([[r["weights"][k] for k in WEIGHT_NAMES] for r in elites], axis=0)
        candidates = []
    best = max(records, key=lambda r: r["objective"])
    chosen = copy.deepcopy(config); chosen["weights"] = best["weights"]
    # Pareto: fill, lower tail and proxy handling cost. Wall-clock caches are excluded.
    def dominates(a, b):
        va = [a["metrics"]["mean_volume_ratio"], a["metrics"]["p10_volume_ratio"], -a["metrics"]["mean_handling_seconds_proxy"]]
        vb = [b["metrics"]["mean_volume_ratio"], b["metrics"]["p10_volume_ratio"], -b["metrics"]["mean_handling_seconds_proxy"]]
        return all(x >= y for x, y in zip(va, vb)) and any(x > y for x, y in zip(va, vb))
    pareto = [i for i, r in enumerate(records) if not any(dominates(other, r) for other in records)]
    sensitivity = []
    for name in WEIGHT_NAMES:
        for factor in (.8, 1.2):
            adjusted = dict(best["weights"]); adjusted[name] *= factor
            planner.config["weights"] = normalized_weights(adjusted)
            metrics = summarize([run_episode(ep, planner) for ep in episodes])
            sensitivity.append(dict(weight=name, factor=factor, objective=validation_objective(metrics), metrics=metrics))
    return chosen, dict(selection="validation completed-episode objective; test never used", records=records,
        selected_index=records.index(best), pareto_indices=pareto, sensitivity=sensitivity,
        tuning_episode_ids=[ep["episode_id"] for ep in episodes])


def paired_report(rows, seed):
    policies = sorted({r["policy"] for r in rows}); by_policy = {p: [r for r in rows if r["policy"] == p] for p in policies}
    report = dict(policies={p: summarize(values) for p, values in by_policy.items()})
    if "ahead" in by_policy and "greedy" in by_policy:
        greedy = {r["episode_id"]: r for r in by_policy["greedy"]}
        grouped = {}
        differences = []
        for row in by_policy["ahead"]:
            delta = row["volume_ratio"]-greedy[row["episode_id"]]["volume_ratio"]
            grouped.setdefault(row["base_group"], []).append(delta); differences.append(delta)
        values = np.asarray([np.mean(v) for v in grouped.values()]); rng = np.random.default_rng(seed)
        boot = [float(np.mean(rng.choice(values, len(values), replace=True))) for _ in range(1000)]
        report["ahead_vs_greedy"] = dict(mean_paired_fill_delta=float(np.mean(differences)),
            win=sum(d > 1e-8 for d in differences), tie=sum(abs(d) <= 1e-8 for d in differences),
            loss=sum(d < -1e-8 for d in differences), group_bootstrap_95ci=list(map(float, np.quantile(boot, [.025, .975]))))
    report["by_scenario"] = {scenario: {p: summarize([r for r in values if r["scenario"] == scenario])
        for p, values in by_policy.items() if any(r["scenario"] == scenario for r in values)} for scenario in MODES}
    return report
