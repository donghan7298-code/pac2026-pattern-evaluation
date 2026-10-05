"""Additional mission regression on held-out inventory groups; never tunes weights."""
import argparse
import hashlib
import json
from pathlib import Path

from .experiment import make_episodes, paired_report, run_episode
from .planner import Planner, ROOT


def runtime_fingerprint():
    paths = [ROOT/n for n in ("planner.py", "scoring.py", "rollout.py", "experiment.py", "contracts.py", "metrics.py", "model.py")]
    paths += [ROOT.parent.parent/"pacdata"/n for n in ("packing.py", "observe.py", "generate.py", "teacher.py")]
    return {str(p.relative_to(ROOT.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    planner = Planner.released()
    episodes = make_episodes([42, 43, 44], "test", planner.config["seed"], cameras=["preview2"])
    rows = []
    for episode in episodes:
        for mode in ("greedy", "ahead"):
            rows.append(run_episode(episode, planner, mode=mode))
        print(f"mission validation {len(rows)}/{len(episodes)*2}", flush=True)
    report = dict(version="mission-integration-0.3.0", purpose="REGRESSION_NOT_WEIGHT_SELECTION",
        base_groups=[42, 43, 44], scenarios=12, camera="preview2", policy_runs=len(rows),
        model_sha256=hashlib.sha256((ROOT/"artifacts/ranker.json").read_bytes()).hexdigest(),
        learned_config_sha256=hashlib.sha256((ROOT/"artifacts/learned_config.json").read_bytes()).hexdigest(),
        runtime_sha256=runtime_fingerprint(),
        limitations=["synthetic single-pallet geometry", "robot execution not simulated", "three independent inventory groups", "serial wall-clock timing, not a realtime guarantee"],
        report=paired_report(rows, planner.config["seed"]), episodes=rows)
    target = Path(args.out); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__": main()
