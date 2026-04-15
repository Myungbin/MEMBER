import argparse
import ast
import json
from pathlib import Path

import torch

from dataloader import DataSet
from dataset_config import DATASET_META, build_experiment_name, resolve_dataset_config
from model import MEMBER
from trainer import Trainer


DEFAULT_CONFIG = {
    "embedding_size": 16,
    "con_s": 0.1,
    "con_us": 0.1,
    "temp_s": 0.6,
    "temp_us": 0.6,
    "gen": 0.5,
    "layers": 2,
    "layers_sg": 4,
    "dropout": 0.2,
    "lambda_s": 0.5,
    "lambda_us": 0.5,
    "data_variant": None,
    "neg_count": 1,
    "neg_edge": 3,
    "if_load_model": False,
    "topk": [10, 20, 50, 100],
    "metrics": ["hit", "ndcg", "recall"],
    "alpha": 1,
    "lr": 0.005,
    "decay": 1e-7,
    "batch_size": 1024,
    "test_batch_size": 1024,
    "min_epoch": 5,
    "epochs": 100,
    "model_path": "./check_point",
    "check_point": "",
    "model_name": None,
    "device": "cpu",
    "setting": "basic",
    "user_activity_split_type": "top_ratio",
    "user_activity_warm_ratio": 0.2,
    "user_activity_pareto_target": 0.8,
    "user_activity_min_warm_users": 1,
    "item_popularity_split_type": "top_ratio",
    "item_popularity_warm_ratio": 0.2,
    "item_popularity_pareto_target": 0.8,
    "item_popularity_min_warm_items": 1,
    "mask_validation": False,
}


def parse_namespace_from_log(log_file):
    log_path = Path(log_file)
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            marker = "Namespace("
            if marker not in line:
                continue

            namespace_expr = line[line.index(marker):].strip()
            expr = ast.parse(namespace_expr, mode="eval").body
            if not isinstance(expr, ast.Call):
                raise ValueError(f"Failed to parse Namespace from: {log_file}")

            parsed = {}
            for keyword in expr.keywords:
                parsed[keyword.arg] = ast.literal_eval(keyword.value)
            return parsed

    raise ValueError(f"Could not find Namespace(...) in log file: {log_file}")
def resolve_checkpoint(explicit_path, checkpoint_dir, args_config, tag):
    if explicit_path:
        checkpoint_path = Path(explicit_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    checkpoint_dir = Path(checkpoint_dir)
    time_token = args_config.get("TIME")
    if not time_token:
        raise ValueError(
            f"TIME is required to auto-resolve the {tag} checkpoint. "
            "Pass --log_file or an explicit checkpoint path."
        )

    candidate_prefixes = []
    for prefix in [args_config.get("data_name"), args_config.get("model_name")]:
        if prefix and prefix not in candidate_prefixes:
            candidate_prefixes.append(prefix)

    candidates = []
    for prefix in candidate_prefixes:
        candidate = checkpoint_dir / f"{prefix}_{tag}_{time_token}.pth"
        if candidate.exists():
            candidates.append(candidate)

    if not candidates:
        candidates = sorted(checkpoint_dir.glob(f"*_{tag}_{time_token}.pth"))

    if not candidates:
        raise FileNotFoundError(
            f"Could not resolve checkpoint for tag={tag}, TIME={time_token} in {checkpoint_dir}"
        )

    if len(candidates) > 1:
        raise FileExistsError(
            f"Multiple checkpoints matched for tag={tag}, TIME={time_token}: "
            f"{[str(candidate) for candidate in candidates]}. "
            "Pass an explicit checkpoint path."
        )

    return candidates[0]


def build_eval_settings(
    behaviors,
    requested_settings,
    include_user_activity_settings=False,
    include_item_popularity_settings=False,
):
    if not requested_settings or requested_settings == ["all"]:
        segment_settings = []
        if include_user_activity_settings:
            segment_settings.extend(["warm", "cold"])
        if include_item_popularity_settings:
            segment_settings.extend(["item_warm", "item_cold"])
        if segment_settings:
            return segment_settings

        settings = ["basic", "seen", "unseen"]
        for behavior in behaviors[:-1]:
            settings.append(f"{behavior}_seen")
            settings.append(f"{behavior}_unseen")
        return settings

    return requested_settings


def load_checkpoint(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)


def build_args(cli_args):
    config = dict(DEFAULT_CONFIG)

    if cli_args.log_file:
        config.update(parse_namespace_from_log(cli_args.log_file))

    if cli_args.data_name:
        config["data_name"] = cli_args.data_name
    if cli_args.data_variant and cli_args.data_path:
        raise ValueError("--data_variant and --data_path cannot be used together.")
    if cli_args.data_variant is not None:
        config["data_variant"] = cli_args.data_variant
        if not cli_args.data_path:
            config.pop("data_path", None)
    if cli_args.data_path is not None:
        config["data_path"] = cli_args.data_path
        config["data_variant"] = None

    if "data_name" not in config:
        raise ValueError("data_name is required. Pass --data_name or --log_file.")

    config = resolve_dataset_config(config)
    if not config.get("model_name"):
        config["model_name"] = build_experiment_name(
            config["data_name"],
            config.get("data_variant"),
            config.get("data_path"),
        )

    if cli_args.device:
        config["device"] = cli_args.device
    if cli_args.test_batch_size is not None:
        config["test_batch_size"] = cli_args.test_batch_size
    if cli_args.topk:
        config["topk"] = cli_args.topk
    if cli_args.metrics:
        config["metrics"] = cli_args.metrics
    if cli_args.model_path:
        config["model_path"] = cli_args.model_path
    config["user_activity_split_type"] = cli_args.user_activity_split_type
    config["user_activity_warm_ratio"] = cli_args.user_activity_warm_ratio
    config["user_activity_pareto_target"] = cli_args.user_activity_pareto_target
    config["user_activity_min_warm_users"] = cli_args.user_activity_min_warm_users
    config["item_popularity_split_type"] = cli_args.item_popularity_split_type
    config["item_popularity_warm_ratio"] = cli_args.item_popularity_warm_ratio
    config["item_popularity_pareto_target"] = cli_args.item_popularity_pareto_target
    config["item_popularity_min_warm_items"] = cli_args.item_popularity_min_warm_items
    config["mask_validation"] = cli_args.mask_validation

    config["if_load_model"] = False
    config["check_point"] = ""
    config["setting"] = "basic"

    return argparse.Namespace(**config)


def evaluate_settings(trainer, dataset, args, split_name, settings):
    results = {}
    for setting in settings:
        eval_dataset, eval_interacts, eval_gt_length = dataset.get_eval_bundle(split_name, setting)
        metric_dict = trainer.evaluate(
            epoch=0,
            test_batch_size=args.test_batch_size,
            dataset=eval_dataset,
            gt_interacts=eval_interacts,
            gt_length=eval_gt_length,
            setting=setting,
            split_name=split_name,
        )

        results[setting] = {
            "user_count": len(eval_interacts),
            "interaction_count": int(eval_gt_length.sum()) if len(eval_gt_length) > 0 else 0,
            "metrics": metric_dict,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default=None, help="Training log file containing Namespace(...).")
    parser.add_argument("--data_name", type=str, default=None, help="tmall, taobao, or jdata.")
    parser.add_argument("--data_variant", type=str, default=None, help="Optional dataset variant name under ./data_variants/{data_name}/.")
    parser.add_argument("--data_path", type=str, default=None, help="Optional explicit dataset path.")
    parser.add_argument("--device", type=str, default="cpu", help="Evaluation device.")
    parser.add_argument("--split", type=str, default="test", choices=["test", "validation"])
    parser.add_argument(
        "--mask_validation",
        action="store_true",
        help="When evaluating the test split, also exclude validation-buy items from the ranking candidates.",
    )
    parser.add_argument(
        "--settings",
        nargs="*",
        default=["all"],
        help=(
            "all, basic, seen, unseen, warm, cold, item_warm, item_cold, "
            "item_popular, item_unpopular, <behavior>_seen, <behavior>_unseen"
        ),
    )
    parser.add_argument("--model_path", type=str, default="./check_point", help="Checkpoint directory.")
    parser.add_argument("--visited_checkpoint", type=str, default=None, help="Explicit path to visited model checkpoint.")
    parser.add_argument("--unvisited_checkpoint", type=str, default=None, help="Explicit path to unvisited model checkpoint.")
    parser.add_argument("--test_batch_size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--topk", nargs="*", type=int, default=None, help="Override top-k values.")
    parser.add_argument("--metrics", nargs="*", default=None, help="Override metrics.")
    parser.add_argument(
        "--include_user_activity_settings",
        action="store_true",
        help="When --settings all is used, evaluate only warm and cold user groups.",
    )
    parser.add_argument(
        "--include_item_popularity_settings",
        action="store_true",
        help="When --settings all is used, evaluate only warm and cold item groups defined by global train-buy popularity.",
    )
    parser.add_argument(
        "--user_activity_split_type",
        type=str,
        default="top_ratio",
        choices=["top_ratio", "pareto"],
        help="How to define warm users from train buy counts within the evaluation split.",
    )
    parser.add_argument(
        "--user_activity_warm_ratio",
        type=float,
        default=0.2,
        help="Warm-user ratio when user_activity_split_type=top_ratio.",
    )
    parser.add_argument(
        "--user_activity_pareto_target",
        type=float,
        default=0.8,
        help="Cold-user share in an 80/20 Pareto-style user split when user_activity_split_type=pareto.",
    )
    parser.add_argument(
        "--user_activity_min_warm_users",
        type=int,
        default=1,
        help="Minimum number of warm users to keep.",
    )
    parser.add_argument(
        "--item_popularity_split_type",
        type=str,
        default="top_ratio",
        choices=["top_ratio", "pareto"],
        help="How to define warm/popular items from global train-buy counts. top_ratio means top 20 percent of train-buy items by count when used with the default warm ratio.",
    )
    parser.add_argument(
        "--item_popularity_warm_ratio",
        type=float,
        default=0.2,
        help="Warm-item ratio when item_popularity_split_type=top_ratio.",
    )
    parser.add_argument(
        "--item_popularity_pareto_target",
        type=float,
        default=0.8,
        help="Cumulative train-buy share captured by warm items when item_popularity_split_type=pareto.",
    )
    parser.add_argument(
        "--item_popularity_min_warm_items",
        type=int,
        default=1,
        help="Minimum number of warm items to keep.",
    )
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save the evaluation results as JSON.")
    cli_args = parser.parse_args()

    args = build_args(cli_args)
    visited_checkpoint = resolve_checkpoint(
        explicit_path=cli_args.visited_checkpoint,
        checkpoint_dir=cli_args.model_path or args.model_path,
        args_config=vars(args),
        tag="visited_model",
    )
    unvisited_checkpoint = resolve_checkpoint(
        explicit_path=cli_args.unvisited_checkpoint,
        checkpoint_dir=cli_args.model_path or args.model_path,
        args_config=vars(args),
        tag="unvisited_model",
    )

    dataset = DataSet(args)

    model_visited = MEMBER(args, dataset, expert_type="visited").to(args.device)
    model_unvisited = MEMBER(args, dataset, expert_type="unvisited").to(args.device)
    load_checkpoint(model_visited, visited_checkpoint, args.device)
    load_checkpoint(model_unvisited, unvisited_checkpoint, args.device)

    trainer = Trainer(model_visited, model_unvisited, dataset, args)
    settings = build_eval_settings(
        args.behaviors,
        cli_args.settings,
        include_user_activity_settings=cli_args.include_user_activity_settings,
        include_item_popularity_settings=cli_args.include_item_popularity_settings,
    )
    results = evaluate_settings(trainer, dataset, args, cli_args.split, settings)

    summary = {
        "split": cli_args.split,
        "mask_validation": args.mask_validation,
        "data_name": args.data_name,
        "data_variant": getattr(args, "data_variant", None),
        "data_path": args.data_path,
        "behaviors": args.behaviors,
        "visited_checkpoint": str(visited_checkpoint),
        "unvisited_checkpoint": str(unvisited_checkpoint),
        "user_activity_summary": dataset.get_user_activity_summary(cli_args.split),
        "item_popularity_summary": dataset.get_item_popularity_summary(cli_args.split),
        "settings": results,
    }

    print(json.dumps(summary, indent=2))

    if cli_args.output_json:
        output_path = Path(cli_args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
