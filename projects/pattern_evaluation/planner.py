"""Observation-in, action proposal-out integration API. Robot validation is external."""
import copy
import json
import math
import time
from pathlib import Path

from .model import Ranker
from .rollout import future_metrics, sample_futures
from .scoring import efficiency_score, feature_vector, normalized_weights, priority, valid_candidates

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
    if any(not isinstance(n, int) or n < 0 or sku not in obs["catalog"] for sku, n in obs["unseen_inventory"].items()):
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

    def _evaluate(self, obs, candidates=None, mode="ahead"):
        if mode not in ("greedy", "current", "teacher", "ranking", "ahead"):
            raise ValueError("Unknown policy mode")
        rows, rejected = valid_candidates(obs, self.config, candidates)
        if not rows: return None, dict(valid_candidates=0, rejected=rejected)
        ood = obs.get("distribution_status") == "OOD" or obs["camera"]["dimension_sigma_m"] > self.config["ood"]["max_sensor_sigma_m"]
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
                row["rank_score"] = efficiency_score(row["static"], predicted[i] if predicted and mode != "current" else blank, self.config["weights"])
        rows.sort(key=lambda r: priority(r["static"], r["rank_score"]), reverse=True)
        # No trained model or OOD: use Full Teacher instead of hiding a weak Top-K fallback.
        selected = rows if mode == "teacher" or (mode == "ahead" and (not predicted or ood)) else rows[:self.config["top_k"]]
        if mode in ("ahead", "teacher"):
            futures = sample_futures(obs, self.config)
            for row in selected:
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
            winner = max(selected, key=lambda r: priority(r["static"], r["score"]))
        else:
            winner = rows[0]; winner["score"] = winner["rank_score"]
            winner["future"] = None
        return winner, dict(valid_candidates=len(rows), detailed_rollouts=len(selected) if mode in ("ahead", "teacher") else 0,
                            rejected=rejected, model_status=self.model_status, ood=ood,
                            future_draws=self.config["future_draws"], future_depth=self.config["future_depth"])

    def plan(self, observation, candidates=None, mode="ahead", allow_buffer=False):
        started = time.perf_counter(); obs = prepare_observation(observation)
        response = dict(schema_version="ahead-action-2.0", request_id=obs["request_id"],
                        robot_feasibility="NOT_CHECKED", requires_robot_validation=True,
                        weights=self.config["weights"])
        box = obs["current_box"]
        if not obs.get("state_verified", False):
            response.update(action=dict(type="HOLD", reason="STATE_UNVERIFIED"))
        elif box and box.get("visual_damage_observed"):
            response.update(action=dict(type="ROUTE_NG", box_id=box["box_id"], reason="DAMAGE"))
        elif box and (not box.get("measurement_valid", True) or not isinstance(box.get("mass_kg"), (float, int)) or not math.isfinite(box["mass_kg"]) or box["mass_kg"] <= 0 or len(box.get("dimensions_m", [])) != 3 or any(not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in box["dimensions_m"])):
            response.update(action=dict(type="REMEASURE", box_id=box["box_id"]))
        elif box and box.get("top_load_capacity_N") is None:
            response.update(action=dict(type="HOLD", reason="LOAD_CAPACITY_UNKNOWN"))
        else:
            winner, diagnostics = self._evaluate(obs, candidates, mode) if box else (None, {})
            if winner:
                response.update(action=dict(type="PLACE_CURRENT", box_id=box["box_id"], candidate=winner["candidate"]),
                    score=winner["score"], static=winner["static"], future=winner["future"], diagnostics=diagnostics)
            elif allow_buffer and obs["buffer_boxes"]:
                choices = []
                for buffered in obs["buffer_boxes"]:
                    buff_obs = dict(obs, current_box=buffered)
                    win, diag = self._evaluate(buff_obs, None, mode)
                    if win: choices.append((win, diag, buffered))
                if choices:
                    win, diag, buffered = max(choices, key=lambda t: priority(t[0]["static"], t[0]["score"]))
                    response.update(action=dict(type="RETRIEVE_BUFFER", box_id=buffered["box_id"], candidate=win["candidate"]),
                        score=win["score"], static=win["static"], future=win["future"], diagnostics=diag)
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
        return response


def to_packing_env_action(result):
    a = result["action"]; kind = a["type"]
    if kind == "PLACE_CURRENT": return dict(type="PLACE", candidate=a["candidate"])
    if kind == "ROUTE_NG": return dict(type="ROUTE_DAMAGE")
    if kind == "REMEASURE": return dict(type="REMEASURE")
    if kind == "PALLET_CLOSE": return dict(type="STOP")
    raise ValueError("Use the buffer-aware state consumer for action: " + kind)
