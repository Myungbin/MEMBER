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
            "Evaluate buy recommendation quality after splitting users into warm/cold groups "
            "for each auxiliary behavior."
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
        "--warm_ratio",
        "--active_ratio",
        dest="warm_ratio",
        type=float,
        default=0.2,
        help=(
            "Top user ratio treated as warm among users with at least one train interaction "
            "for each non-buy behavior. Default: 0.2"
        ),
    )
    parser.add_argument(
        "--min_warm_users",
        "--min_active_users",
        dest="min_warm_users",
        type=int,
        default=1,
        help="Minimum number of warm users to keep for each behavior. Default: 1",
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

    return argparse.Namespace(**config)


def validate_cli_args(cli_args):
    if not 0 < cli_args.warm_ratio < 1:
        raise ValueError("warm_ratio must be in (0, 1).")
    if cli_args.min_warm_users < 0:
        raise ValueError("min_warm_users must be >= 0.")


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


def resolve_warm_user_count(candidate_user_count, warm_ratio, min_warm_users):
    if candidate_user_count == 0:
        return 0

    warm_count = int(round(candidate_user_count * warm_ratio))
    if min_warm_users > 0:
        warm_count = max(warm_count, min_warm_users)
    warm_count = min(warm_count, candidate_user_count)
    return warm_count


def split_warm_and_cold_users(candidate_user_ids, behavior_counts, warm_ratio, min_warm_users):
    candidate_user_ids = np.array(candidate_user_ids, dtype=int)
    if len(candidate_user_ids) == 0:
        return {
            "positive_user_ids": np.array([], dtype=int),
            "zero_user_ids": np.array([], dtype=int),
            "warm_user_ids": np.array([], dtype=int),
            "cold_user_ids": np.array([], dtype=int),
        }

    positive_mask = behavior_counts[candidate_user_ids] > 0
    positive_user_ids = candidate_user_ids[positive_mask]
    zero_user_ids = np.sort(candidate_user_ids[~positive_mask])

    if len(positive_user_ids) == 0:
        return {
            "positive_user_ids": np.array([], dtype=int),
            "zero_user_ids": zero_user_ids,
            "warm_user_ids": np.array([], dtype=int),
            "cold_user_ids": zero_user_ids,
        }

    sort_order = np.lexsort((positive_user_ids, -behavior_counts[positive_user_ids]))
    sorted_positive_user_ids = positive_user_ids[sort_order]
    warm_count = resolve_warm_user_count(len(sorted_positive_user_ids), warm_ratio, min_warm_users)

    warm_user_ids = sorted_positive_user_ids[:warm_count]
    cold_positive_user_ids = sorted_positive_user_ids[warm_count:]
    cold_user_ids = np.concatenate((cold_positive_user_ids, zero_user_ids))

    return {
        "positive_user_ids": positive_user_ids,
        "zero_user_ids": zero_user_ids,
        "warm_user_ids": warm_user_ids,
        "cold_user_ids": cold_user_ids,
    }


def filter_interacts_by_users(interacts, user_ids):
    user_id_set = {int(user_id) for user_id in user_ids}
    return {
        user_id: items
        for user_id, items in interacts.items()
        if int(user_id) in user_id_set
    }


def get_gt_length(interacts):
    return np.array([len(items) for items in interacts.values()], dtype=int)


def build_empty_metrics(topk, metrics):
    empty_metrics = {}
    for k in topk:
        for metric_name in metrics:
            empty_metrics[f"{metric_name}@{k}"] = 0.0
    return empty_metrics


def evaluate_user_group(trainer, dataset, args, interacts, split_name):
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


def build_behavior_summary(candidate_user_ids, positive_user_ids, zero_user_ids, warm_user_ids, cold_user_ids, behavior_counts):
    candidate_user_ids = np.array(candidate_user_ids, dtype=int)
    positive_user_ids = np.array(positive_user_ids, dtype=int)
    zero_user_ids = np.array(zero_user_ids, dtype=int)
    warm_user_ids = np.array(warm_user_ids, dtype=int)
    cold_user_ids = np.array(cold_user_ids, dtype=int)

    warm_train_interaction_count = int(behavior_counts[warm_user_ids].sum()) if len(warm_user_ids) > 0 else 0
    cold_train_interaction_count = int(behavior_counts[cold_user_ids].sum()) if len(cold_user_ids) > 0 else 0
    total_train_interaction_count = warm_train_interaction_count + cold_train_interaction_count

    def summarize_segment(segment_name, user_ids):
        if len(user_ids) == 0:
            return {
                f"{segment_name}_min_train_interaction_count": 0,
                f"{segment_name}_max_train_interaction_count": 0,
                f"{segment_name}_avg_train_interaction_count": 0.0,
            }

        segment_counts = behavior_counts[user_ids]
        return {
            f"{segment_name}_min_train_interaction_count": int(segment_counts.min()),
            f"{segment_name}_max_train_interaction_count": int(segment_counts.max()),
            f"{segment_name}_avg_train_interaction_count": round(float(segment_counts.mean()), 4),
        }

    summary = {
        "candidate_user_count": int(len(candidate_user_ids)),
        "positive_user_count": int(len(positive_user_ids)),
        "zero_user_count": int(len(zero_user_ids)),
        "warm_user_count": int(len(warm_user_ids)),
        "cold_user_count": int(len(cold_user_ids)),
        "warm_user_share": round(
            len(warm_user_ids) / len(candidate_user_ids), 4
        ) if len(candidate_user_ids) > 0 else 0.0,
        "cold_user_share": round(
            len(cold_user_ids) / len(candidate_user_ids), 4
        ) if len(candidate_user_ids) > 0 else 0.0,
        "warm_positive_user_share": round(
            len(warm_user_ids) / len(positive_user_ids), 4
        ) if len(positive_user_ids) > 0 else 0.0,
        "warm_train_interaction_count": warm_train_interaction_count,
        "cold_train_interaction_count": cold_train_interaction_count,
        "warm_train_interaction_share": round(
            warm_train_interaction_count / total_train_interaction_count, 4
        ) if total_train_interaction_count > 0 else 0.0,
    }
    summary.update(summarize_segment("warm", warm_user_ids))
    summary.update(summarize_segment("cold", cold_user_ids))
    return summary


def evaluate_behavior_groups(trainer, dataset, args, split_name, warm_ratio, min_warm_users):
    split_interacts = get_split_interacts(dataset, split_name)
    candidate_user_ids = [int(user_id) for user_id in split_interacts.keys()]

    results = {}
    for behavior in args.behaviors[:-1]:
        behavior_counts = build_behavior_user_counts(dataset, behavior)
        user_groups = split_warm_and_cold_users(
            candidate_user_ids=candidate_user_ids,
            behavior_counts=behavior_counts,
            warm_ratio=warm_ratio,
            min_warm_users=min_warm_users,
        )

        warm_interacts = filter_interacts_by_users(split_interacts, user_groups["warm_user_ids"])
        cold_interacts = filter_interacts_by_users(split_interacts, user_groups["cold_user_ids"])

        results[behavior] = {
            "summary": build_behavior_summary(
                candidate_user_ids=candidate_user_ids,
                positive_user_ids=user_groups["positive_user_ids"],
                zero_user_ids=user_groups["zero_user_ids"],
                warm_user_ids=user_groups["warm_user_ids"],
                cold_user_ids=user_groups["cold_user_ids"],
                behavior_counts=behavior_counts,
            ),
            "segments": {
                "warm": evaluate_user_group(trainer, dataset, args, warm_interacts, split_name),
                "cold": evaluate_user_group(trainer, dataset, args, cold_interacts, split_name),
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
        warm_ratio=cli_args.warm_ratio,
        min_warm_users=cli_args.min_warm_users,
    )

    summary = {
        "split": cli_args.split,
        "mask_validation": args.mask_validation,
        "data_name": args.data_name,
        "target_behavior": args.behaviors[-1],
        "auxiliary_behaviors": args.behaviors[:-1],
        "split_definition": (
            "For each auxiliary behavior, evaluation-split users with at least one train "
            "interaction are ranked by train interaction count. The top warm_ratio users are "
            "warm, while the remaining positive-count users and all zero-count users are cold."
        ),
        "visited_checkpoint": str(visited_checkpoint),
        "unvisited_checkpoint": str(unvisited_checkpoint),
        "warm_ratio": cli_args.warm_ratio,
        "cold_ratio_over_positive_users": round(1 - cli_args.warm_ratio, 4),
        "min_warm_users": cli_args.min_warm_users,
        "positive_only_split": True,
        "zero_count_users_go_to_cold": True,
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
