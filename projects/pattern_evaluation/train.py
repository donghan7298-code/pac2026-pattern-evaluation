"""CPU-only release pipeline. All tuning precedes the held-out comparison."""
import argparse
import json
import os
from pathlib import Path

# Avoid a large BLAS thread pool for these small matrices.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pacdata.io import JsonlWriter, write_json
from .experiment import collect_labels, make_episodes, paired_report, run_episode, tune_weights
from .model import ranking_metrics, train
from .planner import Planner, load_config


def pipeline(out, quick=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out/"benchmark.json").exists(): raise FileExistsError("Use a fresh --out directory")
    config = load_config()
    if quick: config.update(future_draws=3, future_depth=3, top_k=3)
    counts = (3, 2, 2, 2) if quick else (20, 6, 6, 6)
    splits = {}; start = 2
    for name, count in zip(("train", "model_validation", "weight_validation", "test"), counts):
        splits[name] = list(range(start, start+count)); start += count
    # Inventory group IDs, not individual frames, determine every split.
    train_eps = make_episodes(splits["train"], "train", config["seed"],
        modes=["random", "large_last", "heavy_last", "alternating_size", "sku_blocks", "heavy_weak_mix"], cameras=["topview4"])
    val_eps = make_episodes(splits["model_validation"], "model_validation", config["seed"],
        modes=["random", "large_last", "heavy_last", "occlusion_gap"], cameras=["topview8"])
    train_groups = collect_labels(train_eps, config, 3 if quick else 5)
    val_groups = collect_labels(val_eps, config, 3 if quick else 5)
    for name, values in (("train", train_groups), ("validation", val_groups)):
        writer = JsonlWriter(out/f"{name}.labels.jsonl.gz")
        try:
            for value in values: writer.write(value)
        finally: writer.close()
    model, training = train(train_groups, val_groups, config["weights"], epochs=30 if quick else 100)
    model.save(out/"ranker.json"); write_json(out/"training.json", training)
    tuning_eps = []
    modes = ["random", "large_last", "heavy_last", "sku_blocks", "alternating_size", "heavy_weak_mix"]
    for i, group in enumerate(splits["weight_validation"]):
        tuning_eps += make_episodes([group], "weight_validation", config["seed"], modes=[modes[i%len(modes)]], cameras=["topview4"])
    chosen, tuning = tune_weights(tuning_eps, model, config, rounds=1 if quick else 2, trials_per_round=3 if quick else 10)
    write_json(out/"learned_config.json", chosen); write_json(out/"weight_search.json", tuning)
    # No model selection, weight changes or method selection after this point.
    test_eps = make_episodes(splits["test"], "test", config["seed"], cameras=["preview2", "topview8"])
    test_groups = collect_labels([ep for ep in test_eps if ep["scenario"] in ("random", "heavy_last", "large_last") and ep["camera"]["name"] == "topview8"], chosen, 3)
    training["test"] = ranking_metrics(model, test_groups, chosen["weights"], chosen["top_k"])
    write_json(out/"training.json", training)
    results = []
    for policy in ("greedy", "current", "teacher", "ranking", "ahead", "ahead+buffer"):
        planner = Planner(chosen, model, use_model=False)
        for i, episode in enumerate(test_eps):
            results.append(run_episode(episode, planner, "ahead" if policy == "ahead+buffer" else policy, buffer=policy == "ahead+buffer"))
            if (i+1) % 24 == 0: print(f"holdout {policy}: {i+1}/{len(test_eps)}", flush=True)
    compact = [dict(row, decision_ms=[round(v, 3) for v in row["decision_ms"]]) for row in results]
    writer = JsonlWriter(out/"benchmark_episodes.jsonl.gz")
    try:
        for row in compact: writer.write(row)
    finally: writer.close()
    benchmark = paired_report(results, config["seed"])
    benchmark.update(source="SYNTHETIC_DEVELOPMENT_ONLY", test_selection="frozen before test",
        physical_robot_validation="NOT_CHECKED", single_pallet=True, future_teacher="bounded sampled orders + greedy placement; not optimal")
    write_json(out/"benchmark.json", benchmark)
    manifest = dict(seed=config["seed"], quick=quick, group_indices=splits,
        group_split_disjoint=True, train_episodes=len(train_eps), validation_episodes=len(val_eps),
        tuning_episodes=len(tuning_eps), holdout_episodes=len(test_eps), closed_loop_runs=len(results),
        train_states=len(train_groups), validation_states=len(val_groups), test_states=len(test_groups),
        train_candidate_targets=sum(len(g["rows"]) for g in train_groups),
        validation_candidate_targets=sum(len(g["rows"]) for g in val_groups),
        current_candidate_cap=None, teacher_all_generated_valid_candidates=True,
        feature_count=len(model.value["feature_names"]), future_draws=config["future_draws"], future_depth=config["future_depth"])
    write_json(out/"manifest.json", manifest)
    first = next(ep for ep in test_eps if ep["scenario"] == "random")
    demo = run_episode(first, Planner(chosen, model, use_model=False), keep_trace=True)
    from pacdata.observe import observe
    observation = observe(first, dict(processed=0, pallet=first["pallet"], placed=[], remeasured=[], routed=0))
    write_json(out/"sample_observation.json", observation); write_json(out/"sample_result.json", demo["trace"][0])
    write_json(out/"sample_episode_trace.json", demo)
    print(json.dumps(dict(manifest=manifest, policies=benchmark["policies"]), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="projects/pattern_evaluation/artifacts")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(); pipeline(args.out, args.quick)
