import argparse
import csv
import json
from pathlib import Path


DEFAULT_INPUTS = ["./output/eval_user_buy_activity"]
USER_SETTINGS = {"warm", "cold"}
DATASET_ORDER = {"jdata": 0, "taobao": 1, "tmall": 2}
SPLIT_ORDER = {"warm": 0, "cold": 1}


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

        summary = payload.get("user_activity_summary") or {}
        candidate_user_count = int(summary.get("candidate_user_count", 0) or 0)
        warm_user_count = int(summary.get("warm_user_count", 0) or 0)
        cold_user_count = int(summary.get("cold_user_count", 0) or 0)
        warm_train_buy_count = int(summary.get("warm_train_buy_count", 0) or 0)
        cold_train_buy_count = int(summary.get("cold_train_buy_count", 0) or 0)
        total_train_buy_count = warm_train_buy_count + cold_train_buy_count

        for setting, setting_payload in payload.get("settings", {}).items():
            if setting not in USER_SETTINGS:
                continue

            metrics = setting_payload.get("metrics", {})
            for metric_name in metrics:
                if metric_name not in metric_columns:
                    metric_columns.append(metric_name)

            segment_user_count = warm_user_count if setting == "warm" else cold_user_count
            segment_train_buy_count = warm_train_buy_count if setting == "warm" else cold_train_buy_count

            row = {
                "source_file": json_file.name,
                "data_name": payload.get("data_name"),
                "split": payload.get("split"),
                "setting": setting,
                "user_count": setting_payload.get("user_count"),
                "interaction_count": setting_payload.get("interaction_count"),
                "user_split_type": summary.get("split_type"),
                "user_warm_ratio": summary.get("warm_ratio"),
                "user_pareto_target": summary.get("pareto_target"),
                "user_min_warm_users": summary.get("min_warm_users"),
                "candidate_user_count": candidate_user_count,
                "warm_user_count": warm_user_count,
                "cold_user_count": cold_user_count,
                "warm_user_share": summary.get("warm_user_share"),
                "warm_train_buy_count": warm_train_buy_count,
                "cold_train_buy_count": cold_train_buy_count,
                "warm_train_buy_share": summary.get("warm_train_buy_share"),
                "segment_user_count": segment_user_count,
                "segment_user_share": round(
                    segment_user_count / candidate_user_count, 4
                ) if candidate_user_count > 0 else 0.0,
                "segment_train_buy_count": segment_train_buy_count,
                "segment_train_buy_share": round(
                    segment_train_buy_count / total_train_buy_count, 4
                ) if total_train_buy_count > 0 else 0.0,
                "segment_min_train_buy_count": summary.get(f"{setting}_min_train_buy_count"),
                "segment_max_train_buy_count": summary.get(f"{setting}_max_train_buy_count"),
                "segment_avg_train_buy_count": summary.get(f"{setting}_avg_train_buy_count"),
                "visited_checkpoint": payload.get("visited_checkpoint"),
                "unvisited_checkpoint": payload.get("unvisited_checkpoint"),
            }
            row.update(metrics)
            rows.append(row)

    return rows, metric_columns


def write_csv(rows, metric_columns, output_csv):
    fieldnames = [
        "dataset",
        "split",
        "eval_partition",
        "user_count",
        "interaction_count",
        "segment_min_train_buy_count",
        "segment_max_train_buy_count",
        "segment_avg_train_buy_count",
        *metric_columns,
        "user_split_type",
        "user_warm_ratio",
        "user_pareto_target",
        "user_min_warm_users",
        "candidate_user_count",
        "warm_user_count",
        "cold_user_count",
        "warm_user_share",
        "warm_train_buy_count",
        "cold_train_buy_count",
        "warm_train_buy_share",
        "segment_user_count",
        "segment_user_share",
        "segment_train_buy_count",
        "segment_train_buy_share",
        "source_file",
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
        help="JSON files or directories containing user buy activity evaluation outputs.",
    )
    parser.add_argument("--glob", type=str, default="*.json", help="Glob pattern used when an input is a directory.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./output/eval_user_buy_activity/user_buy_activity_eval_summary.csv",
        help="CSV path to write.",
    )
    cli_args = parser.parse_args()

    json_files = collect_json_files(cli_args.inputs, cli_args.glob)
    if not json_files:
        raise FileNotFoundError("No evaluation JSON files were found.")

    rows, metric_columns = build_rows(json_files)
    if not rows:
        raise ValueError("No warm/cold user activity evaluation rows were found in the provided JSON files.")

    for row in rows:
        row["dataset"] = row.pop("data_name")
        row["eval_partition"] = row["split"]
        row["split"] = row.pop("setting")

    rows.sort(
        key=lambda row: (
            DATASET_ORDER.get(row["dataset"], 999),
            SPLIT_ORDER.get(row["split"], 999),
            row["source_file"],
        )
    )
    write_csv(rows, metric_columns, cli_args.output_csv)

    print(f"Wrote {len(rows)} rows to {cli_args.output_csv}")


if __name__ == "__main__":
    main()
