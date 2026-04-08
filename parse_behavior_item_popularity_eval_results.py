import argparse
import csv
import json
from pathlib import Path


DEFAULT_INPUTS = ["./output/eval_behavior_item_popularity"]


def collect_json_files(inputs, pattern):
    json_files = []
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_file():
            if path.suffix.lower() == ".json":
                json_files.append(path)
            continue

        if path.is_dir():
            json_files.extend(sorted(path.glob(pattern)))

    unique_files = []
    seen = set()
    for path in json_files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_files.append(path)

    return unique_files


def build_rows(json_files):
    rows = []
    metric_columns = []

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        for behavior, behavior_payload in payload.get("behavior_results", {}).items():
            summary = behavior_payload.get("summary", {})
            segments = behavior_payload.get("segments", {})

            for item_segment, segment_payload in segments.items():
                metrics = segment_payload.get("metrics", {})
                for metric_name in metrics:
                    if metric_name not in metric_columns:
                        metric_columns.append(metric_name)

                row = {
                    "source_file": json_file.name,
                    "data_name": payload.get("data_name"),
                    "split": payload.get("split"),
                    "target_behavior": payload.get("target_behavior"),
                    "behavior": behavior,
                    "item_segment": item_segment,
                    "popular_ratio": payload.get("popular_ratio"),
                    "unpopular_ratio": payload.get("unpopular_ratio"),
                    "min_popular_items": payload.get("min_popular_items"),
                    "candidate_item_count": summary.get("candidate_item_count"),
                    "train_positive_item_count": summary.get("train_positive_item_count"),
                    "candidate_positive_item_count": summary.get("candidate_positive_item_count"),
                    "candidate_zero_item_count": summary.get("candidate_zero_item_count"),
                    "popular_item_count": summary.get("popular_item_count"),
                    "unpopular_item_count": summary.get("unpopular_item_count"),
                    "popular_item_share": summary.get("popular_item_share"),
                    "popular_positive_item_share": summary.get("popular_positive_item_share"),
                    "zero_count_candidate_item_share": summary.get("zero_count_candidate_item_share"),
                    "popular_train_interaction_count": summary.get("popular_train_interaction_count"),
                    "unpopular_train_interaction_count": summary.get("unpopular_train_interaction_count"),
                    "popular_train_interaction_share": summary.get("popular_train_interaction_share"),
                    "user_count": segment_payload.get("user_count"),
                    "interaction_count": segment_payload.get("interaction_count"),
                    "visited_checkpoint": payload.get("visited_checkpoint"),
                    "unvisited_checkpoint": payload.get("unvisited_checkpoint"),
                }
                row.update(metrics)
                rows.append(row)

    return rows, metric_columns


def write_csv(rows, metric_columns, output_csv):
    fieldnames = [
        "source_file",
        "data_name",
        "split",
        "target_behavior",
        "behavior",
        "item_segment",
        "popular_ratio",
        "unpopular_ratio",
        "min_popular_items",
        "candidate_item_count",
        "train_positive_item_count",
        "candidate_positive_item_count",
        "candidate_zero_item_count",
        "popular_item_count",
        "unpopular_item_count",
        "popular_item_share",
        "popular_positive_item_share",
        "zero_count_candidate_item_share",
        "popular_train_interaction_count",
        "unpopular_train_interaction_count",
        "popular_train_interaction_share",
        "user_count",
        "interaction_count",
        *metric_columns,
        "visited_checkpoint",
        "unvisited_checkpoint",
    ]

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="*",
        default=DEFAULT_INPUTS,
        help="JSON files or directories containing behavior item popularity evaluation outputs.",
    )
    parser.add_argument("--glob", type=str, default="*.json", help="Glob pattern used when an input is a directory.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./output/eval_behavior_item_popularity/behavior_item_popularity_eval_summary.csv",
        help="CSV path to write.",
    )
    cli_args = parser.parse_args()

    json_files = collect_json_files(cli_args.inputs, cli_args.glob)
    if not json_files:
        raise FileNotFoundError("No evaluation JSON files were found.")

    rows, metric_columns = build_rows(json_files)
    if not rows:
        raise ValueError("No behavior item popularity evaluation rows were found in the provided JSON files.")

    rows.sort(
        key=lambda row: (
            row["data_name"],
            row["split"],
            row["behavior"],
            row["item_segment"],
            row["source_file"],
        )
    )
    write_csv(rows, metric_columns, cli_args.output_csv)

    print(f"Wrote {len(rows)} rows to {cli_args.output_csv}")


if __name__ == "__main__":
    main()
