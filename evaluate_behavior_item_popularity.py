import argparse
import json
from pathlib import Path

import numpy as np

from dataloader import DataSet, TestDate
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
            "Evaluate buy recommendation quality after splitting items into "
            "popular/unpopular groups for each non-buy behavior."
        )
    )
    parser.add_argument("--log_file", type=str, default=None, help="Training log file containing Namespace(...).")
    parser.add_argument("--data_name", type=str, default=None, help="tmall, taobao, or jdata.")
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
    parser.add_argument(
        "--popular_ratio",
        type=float,
        default=0.2,
        help=(
            "Top item ratio treated as popular among items with at least one train interaction "
            "for each non-buy behavior. Default: 0.2"
        ),
    )
    parser.add_argument(
        "--min_popular_items",
        type=int,
        default=1,
        help="Minimum number of popular items to keep for each behavior. Default: 1",
    )
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save the evaluation results as JSON.")
    return parser.parse_args()


def build_eval_args(cli_args):
    config = dict(DEFAULT_CONFIG)

    if cli_args.log_file:
        config.update(parse_namespace_from_log(cli_args.log_file))

    if cli_args.data_name:
        config["data_name"] = cli_args.data_name

    if "data_name" not in config:
        raise ValueError("data_name is required. Pass --data_name or --log_file.")

    data_name = config["data_name"].lower()
    if data_name not in DATASET_META:
        raise ValueError(f"Unsupported data_name: {config['data_name']}")

    config["data_name"] = data_name
    config = resolve_dataset_config(config)

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


def validate_cli_args(cli_args):
    if not 0 < cli_args.popular_ratio < 1:
        raise ValueError("popular_ratio must be in (0, 1).")
    if cli_args.min_popular_items < 0:
        raise ValueError("min_popular_items must be >= 0.")


def get_split_interacts(dataset, split_name):
    split_name = split_name.lower()
    if split_name == "test":
        return dataset.test_split_interacts["basic"]
    if split_name == "validation":
        return dataset.validation_split_interacts["basic"]
    raise ValueError(f"Unsupported split: {split_name}")


def build_behavior_item_counts(dataset, behavior):
    behavior_counts = np.zeros(dataset.item_count + 1, dtype=np.int64)
    for items in dataset.train_behavior_dict[behavior].values():
        for item_id in items:
            behavior_counts[int(item_id)] += 1
    return behavior_counts


def collect_candidate_item_ids(interacts):
    candidate_item_ids = sorted({int(item_id) for items in interacts.values() for item_id in items})
    return np.array(candidate_item_ids, dtype=int)


def collect_train_positive_item_ids(behavior_counts):
    positive_item_ids = np.flatnonzero(behavior_counts).astype(int)
    return positive_item_ids[positive_item_ids > 0]


def resolve_popular_item_count(positive_item_count, popular_ratio, min_popular_items):
    if positive_item_count == 0:
        return 0

    popular_count = int(round(positive_item_count * popular_ratio))
    if min_popular_items > 0:
        popular_count = max(popular_count, min_popular_items)
    popular_count = min(popular_count, positive_item_count)
    return popular_count


def sort_item_ids_by_behavior_count(item_ids, behavior_counts):
    item_ids = np.array(item_ids, dtype=int)
    if len(item_ids) == 0:
        return item_ids

    sort_order = np.lexsort((item_ids, -behavior_counts[item_ids]))
    return item_ids[sort_order]


def split_popular_and_unpopular_items(candidate_item_ids, behavior_counts, popular_ratio, min_popular_items):
    candidate_item_ids = np.array(candidate_item_ids, dtype=int)
    train_positive_item_ids = collect_train_positive_item_ids(behavior_counts)

    if len(candidate_item_ids) == 0:
        return {
            "train_positive_item_ids": train_positive_item_ids,
            "candidate_positive_item_ids": np.array([], dtype=int),
            "candidate_zero_item_ids": np.array([], dtype=int),
            "popular_item_ids": np.array([], dtype=int),
            "unpopular_item_ids": np.array([], dtype=int),
        }

    sorted_positive_item_ids = sort_item_ids_by_behavior_count(train_positive_item_ids, behavior_counts)
    popular_count = resolve_popular_item_count(len(sorted_positive_item_ids), popular_ratio, min_popular_items)
    global_popular_item_ids = sorted_positive_item_ids[:popular_count]

    candidate_positive_mask = behavior_counts[candidate_item_ids] > 0
    candidate_positive_item_ids = sort_item_ids_by_behavior_count(candidate_item_ids[candidate_positive_mask], behavior_counts)
    candidate_zero_item_ids = np.sort(candidate_item_ids[~candidate_positive_mask])

    popular_mask = np.isin(candidate_positive_item_ids, global_popular_item_ids, assume_unique=False)
    popular_item_ids = candidate_positive_item_ids[popular_mask]
    unpopular_positive_item_ids = candidate_positive_item_ids[~popular_mask]
    unpopular_item_ids = np.concatenate((unpopular_positive_item_ids, candidate_zero_item_ids))

    return {
        "train_positive_item_ids": train_positive_item_ids,
        "candidate_positive_item_ids": candidate_positive_item_ids,
        "candidate_zero_item_ids": candidate_zero_item_ids,
        "popular_item_ids": popular_item_ids,
        "unpopular_item_ids": unpopular_item_ids,
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


def evaluate_item_group(trainer, dataset, args, interacts, split_name):
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


def build_behavior_summary(
    candidate_item_ids,
    train_positive_item_ids,
    candidate_positive_item_ids,
    candidate_zero_item_ids,
    popular_item_ids,
    unpopular_item_ids,
    behavior_counts,
):
    candidate_item_ids = np.array(candidate_item_ids, dtype=int)
    train_positive_item_ids = np.array(train_positive_item_ids, dtype=int)
    candidate_positive_item_ids = np.array(candidate_positive_item_ids, dtype=int)
    candidate_zero_item_ids = np.array(candidate_zero_item_ids, dtype=int)
    popular_item_ids = np.array(popular_item_ids, dtype=int)
    unpopular_item_ids = np.array(unpopular_item_ids, dtype=int)

    popular_train_interaction_count = int(behavior_counts[popular_item_ids].sum()) if len(popular_item_ids) > 0 else 0
    unpopular_train_interaction_count = (
        int(behavior_counts[unpopular_item_ids].sum()) if len(unpopular_item_ids) > 0 else 0
    )
    total_train_interaction_count = popular_train_interaction_count + unpopular_train_interaction_count

    return {
        "candidate_item_count": int(len(candidate_item_ids)),
        "train_positive_item_count": int(len(train_positive_item_ids)),
        "candidate_positive_item_count": int(len(candidate_positive_item_ids)),
        "candidate_zero_item_count": int(len(candidate_zero_item_ids)),
        "popular_item_count": int(len(popular_item_ids)),
        "unpopular_item_count": int(len(unpopular_item_ids)),
        "popular_item_share": round(
            len(popular_item_ids) / len(candidate_item_ids), 4
        ) if len(candidate_item_ids) > 0 else 0.0,
        "popular_positive_item_share": round(
            len(popular_item_ids) / len(candidate_positive_item_ids), 4
        ) if len(candidate_positive_item_ids) > 0 else 0.0,
        "zero_count_candidate_item_share": round(
            len(candidate_zero_item_ids) / len(candidate_item_ids), 4
        ) if len(candidate_item_ids) > 0 else 0.0,
        "popular_train_interaction_count": popular_train_interaction_count,
        "unpopular_train_interaction_count": unpopular_train_interaction_count,
        "popular_train_interaction_share": round(
            popular_train_interaction_count / total_train_interaction_count, 4
        ) if total_train_interaction_count > 0 else 0.0,
    }


def evaluate_behavior_groups(trainer, dataset, args, split_name, popular_ratio, min_popular_items):
    split_interacts = get_split_interacts(dataset, split_name)
    candidate_item_ids = collect_candidate_item_ids(split_interacts)

    results = {}
    for behavior in args.behaviors[:-1]:
        behavior_counts = build_behavior_item_counts(dataset, behavior)
        item_groups = split_popular_and_unpopular_items(
            candidate_item_ids=candidate_item_ids,
            behavior_counts=behavior_counts,
            popular_ratio=popular_ratio,
            min_popular_items=min_popular_items,
        )

        popular_interacts = filter_interacts_by_items(split_interacts, item_groups["popular_item_ids"])
        unpopular_interacts = filter_interacts_by_items(split_interacts, item_groups["unpopular_item_ids"])

        results[behavior] = {
            "summary": build_behavior_summary(
                candidate_item_ids=candidate_item_ids,
                train_positive_item_ids=item_groups["train_positive_item_ids"],
                candidate_positive_item_ids=item_groups["candidate_positive_item_ids"],
                candidate_zero_item_ids=item_groups["candidate_zero_item_ids"],
                popular_item_ids=item_groups["popular_item_ids"],
                unpopular_item_ids=item_groups["unpopular_item_ids"],
                behavior_counts=behavior_counts,
            ),
            "segments": {
                "popular": evaluate_item_group(trainer, dataset, args, popular_interacts, split_name),
                "unpopular": evaluate_item_group(trainer, dataset, args, unpopular_interacts, split_name),
            },
        }

    return results


def main():
    cli_args = parse_args()
    validate_cli_args(cli_args)
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
        popular_ratio=cli_args.popular_ratio,
        min_popular_items=cli_args.min_popular_items,
    )

    summary = {
        "split": cli_args.split,
        "mask_validation": args.mask_validation,
        "data_name": args.data_name,
        "target_behavior": args.behaviors[-1],
        "auxiliary_behaviors": args.behaviors[:-1],
        "split_definition": (
            "For each auxiliary behavior, items with at least one train interaction are ranked "
            "by train interaction count. The top popular_ratio items are popular, while the "
            "remaining positive-count items and all zero-count evaluation items are unpopular."
        ),
        "visited_checkpoint": str(visited_checkpoint),
        "unvisited_checkpoint": str(unvisited_checkpoint),
        "popular_ratio": cli_args.popular_ratio,
        "unpopular_ratio": round(1 - cli_args.popular_ratio, 4),
        "min_popular_items": cli_args.min_popular_items,
        "positive_only_split": True,
        "zero_count_eval_items_go_to_unpopular": True,
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
