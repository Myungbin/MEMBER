import argparse
import csv
import json
from pathlib import Path


DEFAULT_INPUTS = ["./output/eval_item_popularity"]
DATASET_ORDER = {"jdata": 0, "taobao": 1, "tmall": 2}
SPLIT_ORDER = {"popular": 0, "unpopular": 1}
ITEM_SETTINGS = {
    "item_warm",
    "item_cold",
    "warm_item",
    "cold_item",
    "item_popular",
    "item_unpopular",
    "popular_item",
    "unpopular_item",
    "popular",
    "unpopular",
}


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


def normalize_setting(setting):
    aliases = {
        "item_warm": "popular",
        "warm_item": "popular",
        "item_popular": "popular",
        "popular_item": "popular",
        "popular": "popular",
        "item_cold": "unpopular",
        "cold_item": "unpopular",
        "item_unpopular": "unpopular",
        "unpopular_item": "unpopular",
        "unpopular": "unpopular",
    }
    return aliases.get(setting)


def sort_metric_columns(metric_columns):
    metric_order = {"hit": 0, "ndcg": 1, "recall": 2}

    def metric_key(metric_name):
        name, topk = metric_name.split("@")
        return (int(topk), metric_order.get(name, 999), name)

    return sorted(metric_columns, key=metric_key)


def build_rows(json_files):
    rows = []
    metric_columns = set()

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        for setting, setting_payload in payload.get("settings", {}).items():
            normalized_setting = normalize_setting(setting)
            if normalized_setting is None or setting not in ITEM_SETTINGS:
                continue

            metrics = setting_payload.get("metrics", {})
            for metric_name in metrics:
                metric_columns.add(metric_name)

            row = {
                "dataset": payload.get("data_name"),
                "split": normalized_setting,
                "n": setting_payload.get("user_count"),
                "interaction_count": setting_payload.get("interaction_count"),
            }
            row.update(metrics)
            rows.append(row)

    return rows, sort_metric_columns(metric_columns)


def write_csv(rows, metric_columns, output_csv):
    fieldnames = [
        "dataset",
        "split",
        "n",
        "interaction_count",
        *metric_columns,
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
        help="JSON files or directories containing item-popularity evaluation outputs.",
    )
    parser.add_argument("--glob", type=str, default="*.json", help="Glob pattern used when an input is a directory.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./output/eval_item_popularity/item_popularity_eval_summary.csv",
        help="CSV path to write.",
    )
    cli_args = parser.parse_args()

    json_files = collect_json_files(cli_args.inputs, cli_args.glob)
    if not json_files:
        raise FileNotFoundError("No evaluation JSON files were found.")

    rows, metric_columns = build_rows(json_files)
    if not rows:
        raise ValueError("No item-popularity evaluation rows were found in the provided JSON files.")

    rows.sort(key=lambda row: (DATASET_ORDER.get(row["dataset"], 999), SPLIT_ORDER.get(row["split"], 999)))
    write_csv(rows, metric_columns, cli_args.output_csv)

    print(f"Wrote {len(rows)} rows to {cli_args.output_csv}")


if __name__ == "__main__":
    main()
