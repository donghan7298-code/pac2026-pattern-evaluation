"""Label externally generated patterns using the same safety and future evaluator."""
from .planner import prepare_observation
from .rollout import future_metrics, sample_futures
from .scoring import efficiency_score, feature_vector, priority, valid_candidates


def label_observation(observation, config, candidates=None, base_group="external", split="train"):
    obs = prepare_observation(observation)
    if not obs.get("state_verified") or obs["current_box"] is None:
        raise ValueError("A verified state and a measured current box are required")
    rows, rejected = valid_candidates(obs, config, candidates)
    if not rows: raise ValueError("No valid candidates to label")
    futures = sample_futures(obs, config)
    for row in rows:
        row["features"] = feature_vector(obs, row["candidate"], row["static"])
        row["future"] = future_metrics(obs, row["candidate"], futures, config)
    best = max(range(len(rows)), key=lambda i: priority(rows[i]["static"], efficiency_score(rows[i]["static"], rows[i]["future"], config["weights"])))
    return dict(request_id=obs["request_id"], base_group=base_group, split=split,
                generated_valid_candidates=len(rows), labeled_candidates=len(rows), rows=rows,
                best_candidate_index=best, rejected=rejected,
                teacher="ALL_GENERATED_VALID_CANDIDATES_BOUNDED_GREEDY_FUTURES")
