"""Retrain from external full-candidate labels, or tune from shared-schema episodes."""
import argparse
import random
from pathlib import Path

from pacdata.io import read_jsonl, write_json
from .experiment import tune_weights
from .model import Ranker, train
from .planner import load_config


def require_disjoint(a, b):
    shared = {g["base_group"] for g in a}.intersection(g["base_group"] for g in b)
    if shared: raise ValueError("Inventory groups overlap between splits: " + ", ".join(sorted(shared)))


def select_weight_episodes(episodes, limit, seed):
    """Balance inventory groups and scenarios; camera variants share one case."""
    if limit < 1: raise ValueError("limit must be positive")
    cases = {}
    for episode in episodes:
        if episode.get("split") == "test":
            raise ValueError("Weight fitting must not consume test episodes")
        key = (episode["base_group"], episode["scenario"])
        previous = cases.get(key)
        if previous is None or episode.get("camera", {}).get("name") == "topview4":
            cases[key] = episode
    by_group = {}
    for (group, scenario), episode in sorted(cases.items()):
        by_group.setdefault(group, []).append(episode)
    rng = random.Random(seed)
    groups = sorted(by_group); rng.shuffle(groups)
    for values in by_group.values(): rng.shuffle(values)
    selected = []
    while len(selected) < limit:
        added = False
        for group in groups:
            if by_group[group]:
                selected.append(by_group[group].pop(0)); added = True
                if len(selected) == limit: break
        if not added: break
    if not selected: raise ValueError("Empty tuning dataset")
    return selected


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    rank = sub.add_parser("ranker")
    rank.add_argument("--train", required=True); rank.add_argument("--validation", required=True)
    rank.add_argument("--epochs", type=int, default=100)
    weights = sub.add_parser("weights")
    weights.add_argument("--episodes", required=True); weights.add_argument("--model", required=True)
    weights.add_argument("--limit", type=int, default=12)
    for child in (rank, weights):
        child.add_argument("--config", default="projects/pattern_evaluation/config.json")
        child.add_argument("--out", required=True)
    args = parser.parse_args(); output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    if args.command == "ranker":
        if (output/"ranker.json").exists(): raise FileExistsError("Use a fresh --out directory")
        a = list(read_jsonl(args.train)); b = list(read_jsonl(args.validation)); require_disjoint(a, b)
        model, report = train(a, b, config["weights"], config["seed"], args.epochs)
        model.save(output/"ranker.json"); write_json(output/"training.json", report)
    else:
        if (output/"learned_config.json").exists(): raise FileExistsError("Use a fresh --out directory")
        model = Ranker.load(args.model)
        episodes = select_weight_episodes(read_jsonl(args.episodes), args.limit, config["seed"])
        blocked = set(model.value["training"].get("train_base_groups", []))
        if blocked.intersection(ep["base_group"] for ep in episodes):
            raise ValueError("Weight validation reuses model-training inventory groups")
        chosen, report = tune_weights(episodes, model, config)
        write_json(output/"learned_config.json", chosen); write_json(output/"weight_search.json", report)


if __name__ == "__main__": main()
