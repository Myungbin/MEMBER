import argparse
import csv
import json
from pathlib import Path


DEFAULT_INPUTS = ["./output/eval_behavior_degree_zero"]


ENTITY_TYPE_ALIASES = {
    "user": "user",
    "user_degree": "user",
    "item": "item",
    "item_degree": "item",
}


SEGMENT_ALIASES = {
    "isolated": "isolated",
    "degree_zero": "isolated",
    "connected": "connected",
    "non_degree_zero": "connected",
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


def build_rows(json_files):
    rows = []
    metric_columns = []

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        for behavior, behavior_payload in payload.get("behavior_results", {}).items():
            for raw_entity_type, entity_payload in behavior_payload.items():
                entity_type = ENTITY_TYPE_ALIASES.get(raw_entity_type, raw_entity_type)
                summary = entity_payload.get("summary", {})
                segments = entity_payload.get("segments", {})

                for raw_segment_name, segment_payload in segments.items():
                    segment_name = SEGMENT_ALIASES.get(raw_segment_name, raw_segment_name)
                    metrics = segment_payload.get("metrics", {})
                    for metric_name in metrics:
                        if metric_name not in metric_columns:
                            metric_columns.append(metric_name)

                    candidate_entity_count = summary.get(f"candidate_{entity_type}_count")
                    isolated_entity_count = (
                        summary.get(f"isolated_{entity_type}_count")
                        if summary.get(f"isolated_{entity_type}_count") is not None
                        else summary.get(f"degree_zero_{entity_type}_count")
                    )
                    connected_entity_count = (
                        summary.get(f"connected_{entity_type}_count")
                        if summary.get(f"connected_{entity_type}_count") is not None
                        else summary.get(f"non_degree_zero_{entity_type}_count")
                    )
                    isolated_entity_share = (
                        summary.get(f"isolated_{entity_type}_share")
                        if summary.get(f"isolated_{entity_type}_share") is not None
                        else summary.get(f"degree_zero_{entity_type}_share")
                    )
                    connected_entity_share = (
                        summary.get(f"connected_{entity_type}_share")
                        if summary.get(f"connected_{entity_type}_share") is not None
                        else (
                            round(connected_entity_count / candidate_entity_count, 4)
                            if candidate_entity_count and connected_entity_count is not None
                            else None
                        )
                    )
                    isolated_train_interaction_count = (
                        summary.get("isolated_train_interaction_count")
                        if summary.get("isolated_train_interaction_count") is not None
                        else summary.get("degree_zero_train_interaction_count")
                    )
                    connected_train_interaction_count = (
                        summary.get("connected_train_interaction_count")
                        if summary.get("connected_train_interaction_count") is not None
                        else summary.get("non_degree_zero_train_interaction_count")
                    )
                    n_value = segment_payload.get("user_count")

                    row = {
                        "source_file": json_file.name,
                        "data_name": payload.get("data_name"),
                        "split": payload.get("split"),
                        "target_behavior": payload.get("target_behavior"),
                        "behavior": behavior,
                        "entity_type": entity_type,
                        "segment": segment_name,
                        "candidate_entity_count": candidate_entity_count,
                        "isolated_entity_count": isolated_entity_count,
                        "connected_entity_count": connected_entity_count,
                        "isolated_entity_share": isolated_entity_share,
                        "connected_entity_share": connected_entity_share,
                        "isolated_train_interaction_count": isolated_train_interaction_count,
                        "connected_train_interaction_count": connected_train_interaction_count,
                        "n": n_value,
                        "user_count": n_value,
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
        "entity_type",
        "segment",
        "candidate_entity_count",
        "isolated_entity_count",
        "connected_entity_count",
        "isolated_entity_share",
        "connected_entity_share",
        "isolated_train_interaction_count",
        "connected_train_interaction_count",
        "n",
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
        help="JSON files or directories containing behavior degree-zero evaluation outputs.",
    )
    parser.add_argument("--glob", type=str, default="*.json", help="Glob pattern used when an input is a directory.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./output/eval_behavior_degree_zero/behavior_degree_zero_eval_summary.csv",
        help="CSV path to write.",
    )
    cli_args = parser.parse_args()

    json_files = collect_json_files(cli_args.inputs, cli_args.glob)
    if not json_files:
        raise FileNotFoundError("No evaluation JSON files were found.")

    rows, metric_columns = build_rows(json_files)
    if not rows:
        raise ValueError("No behavior degree-zero evaluation rows were found in the provided JSON files.")

    rows.sort(
        key=lambda row: (
            row["data_name"],
            row["split"],
            row["behavior"],
            row["entity_type"],
            row["segment"],
            row["source_file"],
        )
    )
    write_csv(rows, metric_columns, cli_args.output_csv)

    print(f"Wrote {len(rows)} rows to {cli_args.output_csv}")


if __name__ == "__main__":
    main()
