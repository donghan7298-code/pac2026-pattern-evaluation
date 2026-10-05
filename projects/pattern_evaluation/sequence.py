"""Check an externally proposed ordered pattern using only confirmed box order."""
import copy

from pacdata.packing import apply_placement
from .contracts import preview_box, state_errors
from .metrics import pattern_metrics
from .planner import Planner, prepare_observation


def evaluate_sequence(observation, steps, config=None):
    obs = prepare_observation(observation)
    if not isinstance(steps, list) or not steps:
        raise ValueError("steps must be a nonempty ordered list")
    if not obs.get("state_verified") or state_errors(obs) or obs.get("process_state", "NORMAL") != "NORMAL":
        raise ValueError("Reconcile the observed state before checking a sequence")
    queue = ([copy.deepcopy(obs["current_box"])] if obs["current_box"] else [])
    queue += [preview_box(t) for t in obs["observed_preview"]]
    shelf = {b["box_id"]: copy.deepcopy(b) for b in obs["buffer_boxes"]}
    placed = copy.deepcopy(obs["placed_boxes"]); records = []
    planner = Planner(config, use_model=False)
    result = dict(schema_version="ahead-sequence-1.0", request_id=obs["request_id"],
                  packing_valid=False, requires_observation_refresh_each_step=True, steps=records)
    for index, step in enumerate(steps):
        identity = step["box_id"]
        from_buffer = identity in shelf
        if from_buffer:
            box = shelf[identity]
        elif queue and identity == queue[0]["box_id"]:
            box = queue[0]
        else:
            result.update(status="INVALID_SEQUENCE", failed_step=index,
                          reason="BOX_UNAVAILABLE_OR_CONFIRMED_ORDER_VIOLATION")
            return result
        work = dict(obs, placed_boxes=placed, current_box=box,
                    buffer_boxes=[b for key, b in shelf.items() if key != identity],
                    observed_preview=[], decision_budget_ms=None)
        if index:
            work["state_version"] = str(obs.get("state_version", "unversioned"))+f"/planned:{index}"
        # Score only the declared next placement; no unconfirmed future sequence is executed.
        decision = planner.plan(work, [step["candidate"]], mode="current")
        records.append(dict(step=index, box_id=identity, source="BUFFER" if from_buffer else "CONVEYOR",
                            state_token=decision.get("state_token"), decision=decision))
        if decision["action"]["type"] != "PLACE_CURRENT":
            robot_rejected = any(r.get("stage") == "ROBOT" for r in decision.get("diagnostics", {}).get("rejected", []))
            result.update(status="ROBOT_REPLAN_REQUIRED" if robot_rejected else "INVALID_SEQUENCE",
                          packing_valid=None if robot_rejected else False, failed_step=index,
                          reason=decision["action"].get("reason", decision["action"]["type"]))
            return result
        placed = apply_placement(obs["pallet"], placed, box, decision["action"]["candidate"])
        if from_buffer: del shelf[identity]
        else: queue.pop(0)
    robot_ok = all(r["decision"]["execution_ready"] for r in records)
    result.update(status="EXTERNALLY_VALIDATED_PLAN" if robot_ok else "PACKING_VALID_REQUIRES_ROBOT",
                  packing_valid=True, robot_validated=robot_ok,
                  mean_static_score=sum(r["decision"]["score"] for r in records)/len(records),
                  minimum_safety=min(r["decision"]["static"]["safety"] for r in records),
                  final_metrics=pattern_metrics(obs["pallet"], placed), final_placed_boxes=placed)
    return result
