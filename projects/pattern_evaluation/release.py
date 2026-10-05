"""Resume evaluation from published checkpoints, saving every completed episode."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pacdata.io import JsonlWriter, read_jsonl, write_json
from .experiment import collect_labels, make_episodes, paired_report, run_episode
from .model import Ranker
from .planner import ROOT, Planner, load_config

_MODEL = None
_CONFIG = None


def initialize(model_path, config_path):
    global _MODEL, _CONFIG
    _MODEL = Ranker.load(model_path); _CONFIG = load_config(config_path)


def evaluate_job(job):
    episode, policy = job
    return run_episode(episode, Planner(_CONFIG, _MODEL, use_model=False),
        "ahead" if policy == "ahead+buffer" else policy, buffer=policy == "ahead+buffer")


def label_job(job):
    episode, config = job
    return collect_labels([episode], config, max_states=5)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recover_labels(output, workers):
    config = load_config(ROOT/"config.json")
    for name, indices, modes, camera in (
        ("train", range(2, 22), ["random", "large_last", "heavy_last", "alternating_size", "sku_blocks", "heavy_weak_mix"], "topview4"),
        ("validation", range(22, 28), ["random", "large_last", "heavy_last", "occlusion_gap"], "topview8")):
        target = output/f"{name}.labels.jsonl.gz"
        if target.exists():
            print(f"Keeping existing {target.name}", flush=True); continue
        episodes = make_episodes(indices, "train" if name == "train" else "model_validation",
            config["seed"], modes=modes, cameras=[camera])
        values = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(label_job, (ep, config)): i for i, ep in enumerate(episodes)}
            for future in as_completed(jobs): values[jobs[future]] = future.result()
        writer = JsonlWriter(target)
        try:
            for i in range(len(episodes)):
                for value in values[i]: writer.write(value)
        finally: writer.close()
        states = sum(len(v) for v in values.values())
        candidates = sum(len(g["rows"]) for v in values.values() for g in v)
        print(f"Recovered {name}: {states} states, {candidates} candidates", flush=True)


def evaluate_release(output, workers=1):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model_path = ROOT/"artifacts"/"ranker.json"
    config_path = ROOT/"artifacts"/"learned_config.json"
    config = load_config(config_path); model = Ranker.load(model_path)
    episodes = make_episodes(range(34, 40), "test", config["seed"], cameras=["preview2", "topview8"])
    policies = ["greedy", "current", "teacher", "ranking", "ahead", "ahead+buffer"]
    jobs = [(episode, policy) for episode in episodes for policy in policies]
    frozen = dict(model_sha256=digest(model_path), config_sha256=digest(config_path),
        code_sha256={str(p.relative_to(ROOT.parent.parent)): digest(p) for p in
            [ROOT/"planner.py", ROOT/"scoring.py", ROOT/"rollout.py", ROOT/"experiment.py",
             ROOT.parent.parent/"pacdata"/"packing.py", ROOT.parent.parent/"pacdata"/"observe.py"]},
        episode_ids=[ep["episode_id"] for ep in episodes], policies=policies)
    state_path = output/"evaluation_state.json"; journal = output/"evaluation_progress.jsonl"
    if state_path.exists():
        if json.loads(state_path.read_text())["frozen"] != frozen:
            raise ValueError("Checkpoint differs from existing evaluation. Use another --out directory")
    else: write_json(state_path, dict(frozen=frozen, workers=workers))
    done = {}
    for saved_results in (output/"benchmark_episodes.jsonl.gz", journal):
        if saved_results.exists():
            for row in read_jsonl(saved_results): done[(row["episode_id"], row["policy"])] = row
    pending = [job for job in jobs if (job[0]["episode_id"], job[1]) not in done]
    print(f"Keeping {len(done)} completed episodes; running {len(pending)} remaining", flush=True)
    with journal.open("a", encoding="utf-8") as stream:
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize,
                                 initargs=(str(model_path), str(config_path))) as pool:
            futures = {pool.submit(evaluate_job, job): job for job in pending}
            for future in as_completed(futures):
                row = future.result(); key = (row["episode_id"], row["policy"])
                done[key] = row
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
                stream.flush(); os.fsync(stream.fileno())
                if len(done) % 48 == 0 or len(done) == len(jobs):
                    print(f"Evaluation {len(done)}/{len(jobs)}", flush=True)
    rows = [done[(ep["episode_id"], p)] for ep, p in jobs]
    writer = JsonlWriter(output/"benchmark_episodes.jsonl.gz")
    try:
        for row in rows: writer.write(row)
    finally: writer.close()
    report = paired_report(rows, config["seed"])
    report.update(source="SYNTHETIC_DEVELOPMENT_ONLY", test_selection="fixed groups 34..39; no selection from results",
        physical_robot_validation="NOT_CHECKED", single_pallet=True,
        future_teacher="bounded sampled orders + greedy placement; not optimal",
        benchmark_execution=dict(workers=workers, timing_scope="concurrent CPU evaluation, not deployment latency"),
        checkpoints=dict(model_sha256=frozen["model_sha256"], config_sha256=frozen["config_sha256"]))
    write_json(output/"benchmark.json", report)
    training = json.loads((ROOT/"artifacts"/"training.json").read_text())
    manifest = dict(seed=config["seed"], group_indices=dict(train=list(range(2,22)),
        model_validation=list(range(22,28)), weight_validation=list(range(28,34)), test=list(range(34,40))),
        group_split_disjoint=True, train_episodes=120, validation_episodes=24, tuning_episodes=6,
        holdout_episodes=len(episodes), closed_loop_runs=len(rows), train_states=training["train"]["groups"],
        validation_states=training["validation"]["groups"], test_states=training["test"]["groups"],
        train_candidate_targets=training["train"]["candidates"], validation_candidate_targets=training["validation"]["candidates"],
        teacher_all_generated_valid_candidates=True, current_candidate_cap=None,
        feature_count=len(model.value["feature_names"]), future_draws=config["future_draws"],
        future_depth=config["future_depth"], checkpoint_reused=True)
    write_json(output/"manifest.json", manifest)
    sample_path = output/"sample_episode_trace.json"
    if sample_path.exists(): sample = json.loads(sample_path.read_text())
    else:
        sample = run_episode(episodes[0], Planner(config, model, use_model=False), keep_trace=True)
        write_json(sample_path, sample)
    from pacdata.observe import observe
    obs = observe(episodes[0], dict(processed=0, pallet=episodes[0]["pallet"], placed=[], remeasured=[], routed=0))
    write_json(output/"sample_observation.json", obs); write_json(output/"sample_result.json", sample["trace"][0])
    files = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in output.iterdir()
             if p.is_file() and p.suffix in (".json", ".gz") and p.name != "release_files.json"}
    write_json(output/"release_files.json", files)
    print(json.dumps(dict(policies=report["policies"], paired=report["ahead_vs_greedy"]), indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT/"artifacts"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--labels", action="store_true")
    args = parser.parse_args()
    available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    if not 1 <= args.workers <= available_cpus:
        raise ValueError("workers must be between 1 and available CPUs")
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    if args.labels: recover_labels(output, args.workers)
    evaluate_release(output, args.workers)


if __name__ == "__main__": main()
