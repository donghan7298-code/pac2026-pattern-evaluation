"""Observation-in, action proposal-out integration API. Robot validation is external."""
import copy
import json
import math
import time
from pathlib import Path

from pacdata.packing import apply_placement
from .contracts import buffer_observation, state_errors, valid_measurement, validate_contract
from .metrics import pattern_metrics
from .model import Ranker
from .rollout import future_metrics, sample_futures
from .scoring import efficiency_score, feature_vector, normalized_weights, priority, score_terms, valid_candidates

ROOT = Path(__file__).resolve().parent


def load_config(path=None):
    config = json.loads(Path(path or ROOT/"config.json").read_text(encoding="utf-8"))
    config["weights"] = normalized_weights(config["weights"])
    if not 1 <= config["top_k"] or not 1 <= config["future_draws"] or not 1 <= config["future_depth"] or config["future_candidate_limit"] < 1 or not 0 < config["cvar_fraction"] <= 1:
        raise ValueError("Invalid search configuration")
    if not 0 < config["safety"]["min_cog_edge"] < config["safety"]["target_cog_edge"] <= 1 or not 0 < config["safety"]["target_load_margin"] <= 1:
        raise ValueError("Invalid safety configuration")
    return config


def prepare_observation(value):
    obs = copy.deepcopy(value)
    if "oracle" in obs: raise ValueError("Oracle data is not a policy observation; call observe() first")
    for name in ("pallet", "placed_boxes", "current_box", "unseen_inventory", "catalog"):
        if name not in obs: raise ValueError("Missing observation field: " + name)
    p = obs["pallet"]
    for key in ("length_m", "width_m", "max_height_m", "max_mass_kg"):
        if not isinstance(p.get(key), (float, int)) or not math.isfinite(p[key]) or p[key] <= 0:
            raise ValueError("Invalid pallet: " + key)
    if any(type(n) is not int or n < 0 or sku not in obs["catalog"] for sku, n in obs["unseen_inventory"].items()):
        raise ValueError("Invalid unseen inventory")
    obs.setdefault("observed_preview", []); obs.setdefault("unordered_visible_hints", [])
    obs.setdefault("camera", {}); obs["camera"].setdefault("dimension_sigma_m", 0.)
    obs.setdefault("request_id", "request"); obs.setdefault("buffer_boxes", [])
    identities = [b["box_id"] for b in obs["placed_boxes"]+obs["buffer_boxes"]]
    if obs["current_box"]: identities.append(obs["current_box"]["box_id"])
    if len(set(identities)) != len(identities): raise ValueError("Duplicate placed/buffer box_id")
    for b in obs["placed_boxes"]:
        if "upper_load_N" not in b or "supporter_id" not in b:
            raise ValueError("Placed boxes require verified upper_load_N and supporter_id")
    validate_contract(obs)
    return obs


class Planner:
    def __init__(self, config=None, model=None, use_model=True):
        released_config = ROOT/"artifacts"/"learned_config.json"
        self.config = copy.deepcopy(config or load_config(released_config if released_config.exists() else None))
        self.config["weights"] = normalized_weights(self.config["weights"])
        self.model = model; self.model_status = "PROVIDED" if model else "DISABLED"
        if use_model and model is None:
            path = ROOT/"artifacts"/"ranker.json"
            if path.exists():
                self.model = Ranker.load(path); self.model_status = "TRAINED_ARTIFACT"
            else: self.model_status = "MISSING_FALLBACK"
        self.future_cache = {}; self.cache_limit = 256

    @classmethod
    def released(cls):
        weights_path = ROOT/"artifacts"/"learned_config.json"
        return cls(load_config(weights_path if weights_path.exists() else None))

    def _evaluate(self, obs, candidates=None, mode="ahead", execution_mode="proposal", deadline=None):
        if mode not in ("greedy", "current", "teacher", "ranking", "ahead"):
            raise ValueError("Unknown policy mode")
        rows, rejected = valid_candidates(obs, self.config, candidates, execution_mode)
        if not rows: return None, dict(generated_candidates=len(rejected), valid_candidates=0, detailed_rollouts=0, rejected=rejected)
        ratio = max(d/obs["pallet"][k] for d, k in zip(obs["current_box"]["dimensions_m"], ("length_m", "width_m", "max_height_m")))
        ood = (obs.get("distribution_status") == "OOD" or obs["camera"]["dimension_sigma_m"] > self.config["ood"]["max_sensor_sigma_m"]
               or ratio > self.config["ood"]["max_dimension_ratio"])
        neural = self.model and not ood and mode in ("ahead", "ranking")
        if neural:
            for row in rows: row["features"] = feature_vector(obs, row["candidate"], row["static"])
        predicted = self.model.predict([r["features"] for r in rows]) if neural else None
        blank = dict(mean_fit=0., mean_volume=0., risk=0.)
        if mode == "greedy":
            from pacdata.packing import immediate_score
            for row in rows:
                row["rank_score"] = immediate_score(obs["pallet"], obs["placed_boxes"], obs["current_box"], row["candidate"])
        else:
            for i, row in enumerate(rows):
                row["predicted_future"] = predicted[i] if predicted and mode != "current" else None
                row["rank_score"] = efficiency_score(row["static"], predicted[i] if predicted and mode != "current" else blank, self.config["weights"])
        rows.sort(key=lambda r: priority(r["static"], r["rank_score"]), reverse=True)
        # No trained model or OOD: use Full Teacher instead of hiding a weak Top-K fallback.
        selected = rows if mode == "teacher" or (mode == "ahead" and (not predicted or ood)) else rows[:self.config["top_k"]]
        completed = []; budget_exhausted = False
        if mode in ("ahead", "teacher"):
            futures = sample_futures(obs, self.config)
            for row in selected:
                if deadline is not None and time.perf_counter() >= deadline:
                    budget_exhausted = True; break
                key = json.dumps([self.config["future_candidate_limit"], self.config["cvar_fraction"],
                    self.config["safety"]["min_cog_edge"], obs["pallet"], obs["placed_boxes"], obs["current_box"],
                    row["candidate"], futures], sort_keys=True, separators=(",", ":"))
                future = self.future_cache.get(key)
                if future is None:
                    future = future_metrics(obs, row["candidate"], futures, self.config)
                    if len(self.future_cache) >= self.cache_limit: self.future_cache.pop(next(iter(self.future_cache)))
                    self.future_cache[key] = future
                row["future"] = future
                row["score"] = efficiency_score(row["static"], future, self.config["weights"])
                row["score_future"] = future
                completed.append(row)
            if completed:
                winner = max(completed, key=lambda r: priority(r["static"], r["score"]))
            else:
                winner = rows[0]; winner["score"] = winner["rank_score"]; winner["future"] = None
                winner["score_future"] = winner.get("predicted_future")
        else:
            winner = rows[0]; winner["score"] = winner["rank_score"]
            winner["future"] = None
            winner["score_future"] = winner.get("predicted_future")
        ranking = [dict(candidate=r["candidate"], safety=r["static"]["safety"], rank_score=r["rank_score"],
                        detailed_score=r.get("score") if r in completed else None,
                        robot=r["robot"], selected=r is winner) for r in rows]
        return winner, dict(generated_candidates=len(rows)+len(rejected), valid_candidates=len(rows), detailed_rollouts=len(completed),
                            rejected=rejected, model_status=self.model_status, ood=ood,
                            future_draws=self.config["future_draws"], future_depth=self.config["future_depth"],
                            budget_exhausted=budget_exhausted, ranking=ranking,
                            selection_source="ROLLOUT" if completed else "GREEDY" if mode == "greedy" else "AI_ESTIMATE" if neural else "STATIC_SCORE")

    def _placement_response(self, obs, winner, diagnostics, kind, mode):
        post = apply_placement(obs["pallet"], obs["placed_boxes"], obs["current_box"], winner["candidate"])
        robot = winner["robot"]
        after = pattern_metrics(obs["pallet"], post)
        future = winner["future"]
        return dict(action=dict(type=kind, box_id=obs["current_box"]["box_id"], candidate=winner["candidate"]),
                    score=winner["score"], static=winner["static"], future=future, diagnostics=diagnostics,
                    score_contributions=None if mode == "greedy" else score_terms(winner["static"], winner["score_future"], self.config["weights"]),
                    metrics=dict(before=pattern_metrics(obs["pallet"], obs["placed_boxes"]), after=after,
                        sampled_expected_volume_utilization=after["volume_utilization"]+future["mean_volume"] if future else None),
                    robot=robot, state_token=robot["state_token"], execution_ready=robot["status"] == "PASS",
                    requires_robot_validation=robot["status"] != "PASS",
                    robot_feasibility="EXTERNALLY_VALIDATED" if robot["status"] == "PASS" else "NOT_CHECKED")

    def plan(self, observation, candidates=None, mode="ahead", allow_buffer=False, execution_mode="proposal", buffer_candidates=None):
        started = time.perf_counter(); obs = prepare_observation(observation)
        if execution_mode not in ("proposal", "validated"): raise ValueError("Unknown execution_mode")
        if mode not in ("greedy", "current", "teacher", "ranking", "ahead"): raise ValueError("Unknown policy mode")
        budget = obs.get("decision_budget_ms")
        # A soft budget: packing and safety checks are always completed. Teacher labels never truncate.
        deadline = started+budget/1000 if budget is not None and mode != "teacher" else None
        response = dict(schema_version="ahead-action-2.1", request_id=obs["request_id"],
                        robot_feasibility="NOT_CHECKED", requires_robot_validation=True,
                        execution_ready=False, execution_mode=execution_mode, weights=self.config["weights"])
        box = obs["current_box"]
        invalid_state = state_errors(obs) if obs.get("state_verified", False) else []
        if not obs.get("state_verified", False):
            response.update(action=dict(type="HOLD", reason="STATE_UNVERIFIED"))
        elif invalid_state:
            response.update(action=dict(type="HOLD", reason="STATE_RECONCILIATION_REQUIRED", details=invalid_state))
        elif box and box.get("visual_damage_observed"):
            response.update(action=dict(type="ROUTE_NG", box_id=box["box_id"], reason="DAMAGE"))
        elif box and not valid_measurement(box):
            response.update(action=dict(type="REMEASURE", box_id=box["box_id"]))
        elif box and box.get("top_load_capacity_N") is None:
            response.update(action=dict(type="HOLD", reason="LOAD_CAPACITY_UNKNOWN"))
        elif any(not valid_measurement(b) or b.get("visual_damage_observed") or b.get("top_load_capacity_N") is None for b in obs["buffer_boxes"]):
            response.update(action=dict(type="HOLD", reason="BUFFER_REQUIRES_INSPECTION"))
        elif obs.get("process_state", "NORMAL") != "NORMAL":
            process = obs["process_state"]
            cap = obs.get("buffer_capacity", self.config["buffer"]["capacity"])
            if process == "PALLET_CHANGE" and allow_buffer and box and len(obs["buffer_boxes"]) < cap and box.get("buffer_moves", 0) == 0:
                response.update(action=dict(type="BUFFER_CURRENT", box_id=box["box_id"], buffer_moves=1, reason="PALLET_CHANGE"))
            else:
                response.update(action=dict(type="HOLD", reason=process, conveyor_command="STOP"))
        else:
            winner, diagnostics = self._evaluate(obs, candidates, mode, execution_mode, deadline) if box else (None, {})
            if winner:
                response.update(self._placement_response(obs, winner, diagnostics, "PLACE_CURRENT", mode))
            elif any(r.get("stage") == "ROBOT" for r in diagnostics.get("rejected", [])):
                response.update(action=dict(type="HOLD", reason="ROBOT_VALIDATION_OR_REPLAN_REQUIRED"), diagnostics=diagnostics)
            elif allow_buffer and obs["buffer_boxes"]:
                choices = []
                for buffered in obs["buffer_boxes"]:
                    # Current conveyor box remains pending while a shelf box is placed.
                    buff_obs = buffer_observation(obs, buffered["box_id"])
                    supplied = buffer_candidates.get(buffered["box_id"]) if buffer_candidates is not None else None
                    win, diag = self._evaluate(buff_obs, supplied, mode, execution_mode, deadline)
                    if win: choices.append((win, diag, buff_obs))
                    elif any(r.get("stage") == "ROBOT" for r in diag.get("rejected", [])):
                        response.update(action=dict(type="HOLD", reason="BUFFER_ROBOT_VALIDATION_REQUIRED"), diagnostics=diag)
                if choices:
                    win, diag, buff_obs = max(choices, key=lambda t: priority(t[0]["static"], t[0]["score"]))
                    response.update(self._placement_response(buff_obs, win, diag, "RETRIEVE_BUFFER", mode))
            if "action" not in response:
                cap = obs.get("buffer_capacity", self.config["buffer"]["capacity"])
                if allow_buffer and box and len(obs["buffer_boxes"]) < cap and box.get("buffer_moves", 0) == 0:
                    response.update(action=dict(type="BUFFER_CURRENT", box_id=box["box_id"], buffer_moves=1))
                elif box or obs["buffer_boxes"]:
                    response.update(action=dict(type="PALLET_CLOSE", reason="NO_VALID_FINITE_SEARCH_CANDIDATE"), diagnostics=diagnostics)
                elif sum(obs["unseen_inventory"].values()) or not obs.get("stream_terminated", False):
                    response.update(action=dict(type="WAIT", reason="EXPECTED_UNSEEN"))
                else: response.update(action=dict(type="DONE"))
        response["decision_ms"] = (time.perf_counter()-started)*1000
        response["timing"] = dict(evaluation_ms=response["decision_ms"], upstream_pattern_generation_ms=obs.get("pattern_generation_ms"),
            combined_planning_ms=response["decision_ms"]+obs["pattern_generation_ms"] if "pattern_generation_ms" in obs else None,
            robot_cycle_seconds_estimate=response.get("static", {}).get("robot_cycle_seconds_estimate"),
            robot_cycle_seconds_proxy=response.get("static", {}).get("handling_seconds_proxy"),
            budget_ms=budget, budget_exceeded=budget is not None and response["decision_ms"] > budget,
            budget_kind="SOFT_SAFETY_CHECKS_NEVER_SKIPPED")
        return response


def to_packing_env_action(result):
    a = result["action"]; kind = a["type"]
    if kind == "PLACE_CURRENT": return dict(type="PLACE", candidate=a["candidate"])
    if kind == "ROUTE_NG": return dict(type="ROUTE_DAMAGE")
    if kind == "REMEASURE": return dict(type="REMEASURE")
    if kind == "PALLET_CLOSE": return dict(type="STOP")
    raise ValueError("Use the buffer-aware state consumer for action: " + kind)
