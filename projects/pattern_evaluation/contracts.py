"""Validation at the perception / packing / robot planning boundaries."""
import hashlib
import copy
import json
import math

from pacdata.packing import apply_placement

ROBOT_CHECKS = ("reachability", "ik", "collision_free", "payload",
                "grasp_direction", "approach_pose")


def number(value, minimum=0., positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > minimum if positive else value >= minimum))


def valid_measurement(box):
    return (box.get("measurement_valid", True) is True
            and number(box.get("mass_kg"), positive=True)
            and isinstance(box.get("dimensions_m"), (list, tuple))
            and len(box["dimensions_m"]) == 3
            and all(number(v, positive=True) for v in box["dimensions_m"]))


def validate_box(box, name, measured=True):
    if not isinstance(box, dict) or not isinstance(box.get("box_id"), str) or not box["box_id"]:
        raise ValueError(name + " requires a nonempty box_id")
    if not isinstance(box.get("sku"), str):
        raise ValueError(name + " requires sku")
    allowed = box.get("allowed_yaw_deg")
    if not isinstance(allowed, (list, tuple)) or not allowed or any(type(a) not in (int, float) or a not in (0, 90) for a in allowed):
        raise ValueError(name + " supports upright yaw 0/90 only; convert orientation explicitly")
    capacity = box.get("top_load_capacity_N")
    if capacity is not None and not number(capacity):
        raise ValueError(name + " has invalid top_load_capacity_N")
    if measured and not valid_measurement(box):
        raise ValueError(name + " requires verified positive dimensions and mass")
    moves = box.get("buffer_moves", 0)
    if type(moves) is not int or not 0 <= moves <= 1:
        raise ValueError(name + " buffer_moves must be 0 or 1")


def validate_contract(obs):
    """Schema errors are rejected; damaged/unmeasured current boxes remain routable."""
    if any(not number(obs["pallet"].get(k), positive=True) for k in ("length_m", "width_m", "max_height_m", "max_mass_kg")):
        raise ValueError("Pallet dimensions and mass capacity must be positive numbers")
    if obs.get("coordinate_frame", "pallet") != "pallet":
        raise ValueError("Transform positions to the pallet frame before evaluation")
    if obs.get("units", {"length": "m", "mass": "kg", "force": "N"}) != {"length": "m", "mass": "kg", "force": "N"}:
        raise ValueError("Expected units: m, kg, N")
    if type(obs.get("state_verified", False)) is not bool:
        raise ValueError("state_verified must be boolean")
    if not number(obs["camera"]["dimension_sigma_m"]):
        raise ValueError("Invalid dimension_sigma_m")
    if not number(obs.get("pattern_generation_ms", 0)):
        raise ValueError("Invalid upstream pattern_generation_ms")
    if obs.get("decision_budget_ms") is not None and not number(obs["decision_budget_ms"], positive=True):
        raise ValueError("decision_budget_ms must be positive")
    if type(obs.get("buffer_capacity", 2)) is not int or obs.get("buffer_capacity", 2) < len(obs["buffer_boxes"]):
        raise ValueError("Invalid buffer capacity / occupancy")
    if obs.get("process_state", "NORMAL") not in ("NORMAL", "PALLET_CHANGE", "REPACKING", "HOLD"):
        raise ValueError("Unknown process_state")
    for b in obs["placed_boxes"]:
        validate_box(b, "placed box")
        if not isinstance(b.get("position_m"), (list, tuple)) or len(b["position_m"]) != 3 or any(not number(v) for v in b["position_m"]):
            raise ValueError("Invalid placed position_m")
        if b.get("yaw_deg") not in b["allowed_yaw_deg"] or not number(b.get("upper_load_N")):
            raise ValueError("Invalid placed yaw / upper_load_N")
    for b in obs["buffer_boxes"]:
        validate_box(b, "buffer box", measured=False)
        if b.get("buffer_moves") != 1:
            raise ValueError("Buffer boxes require buffer_moves=1")
    if obs["current_box"] is not None:
        validate_box(obs["current_box"], "current box", measured=False)
    for sku, item in obs["catalog"].items():
        validate_box(dict(item, box_id="catalog:"+sku, sku=sku), "catalog")
        if item.get("top_load_capacity_N") is None:
            raise ValueError("Catalog load capacity is required for future simulation")
    preview_ids = []
    for track in obs["observed_preview"]:
        b = preview_box(track)
        validate_box(b, "preview")
        if b["top_load_capacity_N"] is None or not number(track.get("dimension_sigma_m", 0)):
            raise ValueError("Invalid preview load capacity / uncertainty")
        preview_ids.append(b["box_id"])
    ids = [b["box_id"] for b in obs["placed_boxes"]+obs["buffer_boxes"]]
    if obs["current_box"]: ids.append(obs["current_box"]["box_id"])
    if len(set(ids+preview_ids)) != len(ids+preview_ids):
        raise ValueError("Duplicate box / preview ID")
    profile = obs.get("robot_profile")
    if profile is not None:
        if not number(profile.get("max_payload_kg"), positive=True) or not number(profile.get("eoat_mass_kg")):
            raise ValueError("Robot profile requires max_payload_kg and eoat_mass_kg")


def preview_box(track):
    return dict(box_id=track["track_id"], sku=track["sku"], dimensions_m=list(track["dimensions_m"]),
                mass_kg=track["nominal_mass_kg"], top_load_capacity_N=track["top_load_capacity_N"],
                allowed_yaw_deg=list(track["allowed_yaw_deg"]), measurement_valid=True,
                visual_damage_observed=False)


def buffer_observation(obs, box_id):
    """Context for shelf-to-pallet evaluation; preserve the waiting conveyor box."""
    value = copy.deepcopy(obs)
    buffered = next(b for b in value["buffer_boxes"] if b["box_id"] == box_id)
    current = value["current_box"]
    value["buffer_boxes"] = [b for b in value["buffer_boxes"] if b["box_id"] != box_id]
    value["current_box"] = buffered
    value["pending_conveyor_box"] = current
    if current:
        track = dict(track_id=current["box_id"], sku=current["sku"], dimensions_m=current["dimensions_m"],
                     nominal_mass_kg=current["mass_kg"], top_load_capacity_N=current["top_load_capacity_N"],
                     allowed_yaw_deg=current["allowed_yaw_deg"], dimension_sigma_m=0.)
        value["observed_preview"] = [track]+value["observed_preview"]
    return value


def state_errors(obs):
    """Rebuild support and loads, including after pallet resizing or sensor correction."""
    pending = list(obs["placed_boxes"]); rebuilt = []; by_id = {}
    while pending:
        ready = [b for b in pending if b["supporter_id"] == "FLOOR" or b["supporter_id"] in by_id]
        if not ready: return ["MISSING_SUPPORTER_OR_SUPPORT_CYCLE"]
        for box in ready:
            if box.get("visual_damage_observed") or box.get("top_load_capacity_N") is None:
                return ["PLACED_BOX_REQUIRES_INSPECTION"]
            try:
                rebuilt = apply_placement(obs["pallet"], rebuilt, box,
                                          dict(position_m=box["position_m"], yaw_deg=box["yaw_deg"]))
            except (ValueError, KeyError) as error:
                return ["PLACED_STATE_INVALID: " + str(error)]
            if rebuilt[-1]["supporter_id"] != box["supporter_id"]:
                return ["SUPPORTER_MISMATCH"]
            by_id[box["box_id"]] = box
            pending.remove(box)
    computed = {b["box_id"]: b for b in rebuilt}
    if any(abs(b["upper_load_N"]-computed[b["box_id"]]["upper_load_N"]) > 1e-5 for b in obs["placed_boxes"]):
        return ["UPPER_LOAD_MISMATCH"]
    return []


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def placement_token(candidate):
    return _digest({k: candidate.get(k) for k in ("position_m", "yaw_deg", "grasp_id", "approach_id")})


def state_token(obs):
    # Include the external scene/version: a changed obstacle must invalidate old IK/path results.
    return _digest({k: obs.get(k) for k in ("pallet", "placed_boxes", "current_box", "buffer_boxes",
                    "state_version", "robot_scene_version", "robot_profile", "process_state", "pending_conveyor_box")})


def robot_assessment(obs, candidate):
    """External attestations are context-bound; no IK or collision solver is implied."""
    result = dict(status="NOT_CHECKED", reasons=[], checks={}, estimated_cycle_seconds=None,
                  state_token=state_token(obs), placement_token=placement_token(candidate))
    profile = obs.get("robot_profile")
    if profile and obs["current_box"]["mass_kg"]+profile["eoat_mass_kg"] > profile["max_payload_kg"]+1e-8:
        result.update(status="REJECT", reasons=["ROBOT_PAYLOAD_WITH_EOAT"])
        return result
    feedback = candidate.get("robot_validation")
    if feedback is None: return result
    if not isinstance(feedback, dict): raise ValueError("robot_validation must be an object")
    matched = (feedback.get("state_token") == result["state_token"]
               and feedback.get("placement_token") == result["placement_token"]
               and feedback.get("box_id") == obs["current_box"]["box_id"])
    if not matched:
        result.update(status="STALE", reasons=["ROBOT_FEEDBACK_CONTEXT_MISMATCH"])
        return result
    checks = feedback.get("checks", {})
    if not isinstance(checks, dict) or any(k not in ROBOT_CHECKS or type(v) is not bool for k, v in checks.items()):
        raise ValueError("Robot checks must be named boolean results")
    seconds = feedback.get("estimated_cycle_seconds")
    if seconds is not None and not number(seconds, positive=True):
        raise ValueError("estimated_cycle_seconds must be positive seconds")
    rejected = [k.upper() for k, value in checks.items() if value is False]
    # A version supplied by the scene owner is mandatory for an executable attestation.
    versioned = obs.get("state_version") is not None and obs.get("robot_scene_version") is not None
    complete = all(checks.get(k) is True for k in ROBOT_CHECKS) and versioned
    result.update(status="REJECT" if rejected else "PASS" if complete else "PARTIAL",
                  reasons=rejected or ([] if complete else ["ROBOT_CHECKS_OR_SCENE_VERSION_MISSING"]),
                  checks=checks, estimated_cycle_seconds=seconds, validator=feedback.get("validator"))
    return result
