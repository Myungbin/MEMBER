import argparse
import csv
import json
from pathlib import Path


DEFAULT_INPUTS = ["./output/eval"]


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

        for setting, setting_payload in payload.get("settings", {}).items():
            metrics = setting_payload.get("metrics", {})
            for metric_name in metrics:
                if metric_name not in metric_columns:
                    metric_columns.append(metric_name)

            row = {
                "source_file": json_file.name,
                "data_name": payload.get("data_name"),
                "split": payload.get("split"),
                "setting": setting,
                "user_count": setting_payload.get("user_count"),
                "interaction_count": setting_payload.get("interaction_count"),
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
        "setting",
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
        help="JSON files or directories containing evaluation JSON outputs.",
    )
    parser.add_argument("--glob", type=str, default="*.json", help="Glob pattern used when an input is a directory.")
    parser.add_argument("--output_csv", type=str, default="./output/eval/eval_summary.csv", help="CSV path to write.")
    cli_args = parser.parse_args()

    json_files = collect_json_files(cli_args.inputs, cli_args.glob)
    if not json_files:
        raise FileNotFoundError("No evaluation JSON files were found.")

    rows, metric_columns = build_rows(json_files)
    rows.sort(key=lambda row: (row["data_name"], row["split"], row["setting"], row["source_file"]))
    write_csv(rows, metric_columns, cli_args.output_csv)

    print(f"Wrote {len(rows)} rows to {cli_args.output_csv}")


if __name__ == "__main__":
    main()
