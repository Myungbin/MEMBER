import argparse
import json
from pathlib import Path

import numpy as np

from dataloader import DataSet, TestDate
from dataset_config import build_experiment_name
from evaluate_saved_model import (
    DEFAULT_CONFIG,
    DATASET_META,
    load_checkpoint,
    parse_namespace_from_log,
    resolve_checkpoint,
    resolve_dataset_config,
)
from model import MEMBER
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate buy recommendation quality for users and items whose train degree "
            "is zero under each behavior, including buy."
        )
    )
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
    parser.add_argument("--model_path", type=str, default="./check_point", help="Checkpoint directory.")
    parser.add_argument("--visited_checkpoint", type=str, default=None, help="Explicit path to visited model checkpoint.")
    parser.add_argument("--unvisited_checkpoint", type=str, default=None, help="Explicit path to unvisited model checkpoint.")
    parser.add_argument("--test_batch_size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--topk", nargs="*", type=int, default=None, help="Override top-k values.")
    parser.add_argument("--metrics", nargs="*", default=None, help="Override metrics.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save the evaluation results as JSON.")
    return parser.parse_args()


def build_eval_args(cli_args):
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

    data_name = config["data_name"].lower()
    if data_name not in DATASET_META:
        raise ValueError(f"Unsupported data_name: {config['data_name']}")

    config["data_name"] = data_name
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
    config["mask_validation"] = cli_args.mask_validation

    config["if_load_model"] = False
    config["check_point"] = ""
    config["setting"] = "basic"
    config.setdefault("TIME", "eval")

    return argparse.Namespace(**config)


def get_split_interacts(dataset, split_name):
    split_name = split_name.lower()
    if split_name == "test":
        return dataset.test_split_interacts["basic"]
    if split_name == "validation":
        return dataset.validation_split_interacts["basic"]
    raise ValueError(f"Unsupported split: {split_name}")


def build_behavior_user_counts(dataset, behavior):
    behavior_counts = np.zeros(dataset.user_count + 1, dtype=np.int64)
    for user_id, items in dataset.train_behavior_dict[behavior].items():
        behavior_counts[int(user_id)] = len(items)
    return behavior_counts


def build_behavior_item_counts(dataset, behavior):
    behavior_counts = np.zeros(dataset.item_count + 1, dtype=np.int64)
    for items in dataset.train_behavior_dict[behavior].values():
        for item_id in items:
            behavior_counts[int(item_id)] += 1
    return behavior_counts


def collect_candidate_user_ids(interacts):
    return np.array(sorted(int(user_id) for user_id in interacts.keys()), dtype=int)


def collect_candidate_item_ids(interacts):
    return np.array(sorted({int(item_id) for items in interacts.values() for item_id in items}), dtype=int)


def split_degree_zero_ids(candidate_ids, degree_counts):
    candidate_ids = np.array(candidate_ids, dtype=int)
    if len(candidate_ids) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    zero_mask = degree_counts[candidate_ids] == 0
    degree_zero_ids = candidate_ids[zero_mask]
    non_degree_zero_ids = candidate_ids[~zero_mask]
    return degree_zero_ids, non_degree_zero_ids


def filter_interacts_by_users(interacts, user_ids):
    user_id_set = {int(user_id) for user_id in user_ids}
    return {
        user_id: items
        for user_id, items in interacts.items()
        if int(user_id) in user_id_set
    }


def filter_interacts_by_items(interacts, item_ids):
    item_id_set = {int(item_id) for item_id in item_ids}
    filtered_interacts = {}

    for user_id, items in interacts.items():
        filtered_items = [item_id for item_id in items if int(item_id) in item_id_set]
        if filtered_items:
            filtered_interacts[user_id] = filtered_items

    return filtered_interacts


def get_gt_length(interacts):
    return np.array([len(items) for items in interacts.values()], dtype=int)


def build_empty_metrics(topk, metrics):
    empty_metrics = {}
    for k in topk:
        for metric_name in metrics:
            empty_metrics[f"{metric_name}@{k}"] = 0.0
    return empty_metrics


def evaluate_interacts(trainer, dataset, args, interacts, split_name):
    if not interacts:
        return {
            "user_count": 0,
            "interaction_count": 0,
            "metrics": build_empty_metrics(args.topk, args.metrics),
        }

    user_samples = [int(user_id) for user_id in interacts.keys()]
    eval_dataset = TestDate(dataset.user_count, dataset.item_count, samples=user_samples)
    gt_length = get_gt_length(interacts)
    metrics = trainer.evaluate(
        epoch=0,
        test_batch_size=args.test_batch_size,
        dataset=eval_dataset,
        gt_interacts=interacts,
        gt_length=gt_length,
        setting="basic",
        split_name=split_name,
    )
    return {
        "user_count": len(interacts),
        "interaction_count": int(gt_length.sum()),
        "metrics": metrics,
    }


def build_id_group_summary(candidate_ids, degree_zero_ids, non_degree_zero_ids, degree_counts, entity_name):
    candidate_ids = np.array(candidate_ids, dtype=int)
    degree_zero_ids = np.array(degree_zero_ids, dtype=int)
    non_degree_zero_ids = np.array(non_degree_zero_ids, dtype=int)

    isolated_count = int(len(degree_zero_ids))
    connected_count = int(len(non_degree_zero_ids))
    total_count = int(len(candidate_ids))
    connected_train_interaction_count = (
        int(degree_counts[non_degree_zero_ids].sum()) if connected_count > 0 else 0
    )

    return {
        f"candidate_{entity_name}_count": total_count,
        f"isolated_{entity_name}_count": isolated_count,
        f"connected_{entity_name}_count": connected_count,
        f"isolated_{entity_name}_share": round(
            isolated_count / total_count, 4
        ) if total_count > 0 else 0.0,
        f"connected_{entity_name}_share": round(
            connected_count / total_count, 4
        ) if total_count > 0 else 0.0,
        "isolated_train_interaction_count": 0,
        "connected_train_interaction_count": connected_train_interaction_count,
        # Backward-compatible aliases for older parsers.
        f"degree_zero_{entity_name}_count": isolated_count,
        f"non_degree_zero_{entity_name}_count": connected_count,
        f"degree_zero_{entity_name}_share": round(
            isolated_count / total_count, 4
        ) if total_count > 0 else 0.0,
        "degree_zero_train_interaction_count": 0,
        "non_degree_zero_train_interaction_count": connected_train_interaction_count,
    }


def evaluate_behavior_groups(trainer, dataset, args, split_name):
    split_interacts = get_split_interacts(dataset, split_name)
    candidate_user_ids = collect_candidate_user_ids(split_interacts)
    candidate_item_ids = collect_candidate_item_ids(split_interacts)

    results = {}
    for behavior in args.behaviors:
        user_degree_counts = build_behavior_user_counts(dataset, behavior)
        isolated_user_ids, connected_user_ids = split_degree_zero_ids(candidate_user_ids, user_degree_counts)

        item_degree_counts = build_behavior_item_counts(dataset, behavior)
        isolated_item_ids, connected_item_ids = split_degree_zero_ids(candidate_item_ids, item_degree_counts)

        results[behavior] = {
            "user": {
                "summary": build_id_group_summary(
                    candidate_ids=candidate_user_ids,
                    degree_zero_ids=isolated_user_ids,
                    non_degree_zero_ids=connected_user_ids,
                    degree_counts=user_degree_counts,
                    entity_name="user",
                ),
                "segments": {
                    "isolated": evaluate_interacts(
                        trainer,
                        dataset,
                        args,
                        filter_interacts_by_users(split_interacts, isolated_user_ids),
                        split_name,
                    ),
                    "connected": evaluate_interacts(
                        trainer,
                        dataset,
                        args,
                        filter_interacts_by_users(split_interacts, connected_user_ids),
                        split_name,
                    ),
                },
            },
            "item": {
                "summary": build_id_group_summary(
                    candidate_ids=candidate_item_ids,
                    degree_zero_ids=isolated_item_ids,
                    non_degree_zero_ids=connected_item_ids,
                    degree_counts=item_degree_counts,
                    entity_name="item",
                ),
                "segments": {
                    "isolated": evaluate_interacts(
                        trainer,
                        dataset,
                        args,
                        filter_interacts_by_items(split_interacts, isolated_item_ids),
                        split_name,
                    ),
                    "connected": evaluate_interacts(
                        trainer,
                        dataset,
                        args,
                        filter_interacts_by_items(split_interacts, connected_item_ids),
                        split_name,
                    ),
                },
            },
        }

    return results


def main():
    cli_args = parse_args()
    args = build_eval_args(cli_args)

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
    behavior_results = evaluate_behavior_groups(
        trainer=trainer,
        dataset=dataset,
        args=args,
        split_name=cli_args.split,
    )

    summary = {
        "split": cli_args.split,
        "mask_validation": args.mask_validation,
        "data_name": args.data_name,
        "data_variant": getattr(args, "data_variant", None),
        "data_path": args.data_path,
        "target_behavior": args.behaviors[-1],
        "evaluated_behaviors": args.behaviors,
        "split_definition": (
            "For each behavior, connected means appearing at least once in the corresponding train "
            "behavior data and isolated means never appearing. This is evaluated independently for "
            "users and items, including buy."
        ),
        "n_definition": {
            "user": "Number of evaluation-split users in the segment.",
            "item": "Number of evaluation-split users with at least one GT item in the segment.",
        },
        "visited_checkpoint": str(visited_checkpoint),
        "unvisited_checkpoint": str(unvisited_checkpoint),
        "behavior_results": behavior_results,
    }

    print(json.dumps(summary, indent=2))

    if cli_args.output_json:
        output_path = Path(cli_args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
