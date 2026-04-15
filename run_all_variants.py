import argparse
import shlex
import subprocess
import sys

from dataset_config import DATASET_META, get_available_variants


README_PRESET_ARGS = {
    "tmall": [
        "--con_s", "0.1",
        "--temp_s", "0.6",
        "--con_us", "0.1",
        "--temp_us", "0.7",
        "--gen", "0.1",
        "--lambda_s", "0.6",
        "--alpha", "2",
        "--device", "cuda:1",
        "--mask_validation",
    ],
    "taobao": [
        "--con_s", "0.1",
        "--temp_s", "0.8",
        "--con_us", "0.1",
        "--temp_us", "0.7",
        "--gen", "0.1",
        "--lambda_us", "0.6",
        "--device", "cuda:2",
        "--mask_validation",
    ],
    "jdata": [
        "--con_s", "0.1",
        "--temp_s", "0.6",
        "--con_us", "0.01",
        "--temp_us", "1.0",
        "--gen", "0.01",
        "--lambda_s", "0.4",
        "--lambda_us", "0.4",
        "--alpha", "2",
        "--device", "cuda:3",
        "--mask_validation",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MEMBER sequentially for every available data variant."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=sorted(DATASET_META.keys()),
        help="Datasets to run. Default: all available datasets.",
    )
    parser.add_argument(
        "--include_base",
        action="store_true",
        help="Also run the original base dataset without --data_variant.",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="main.py",
        help="Python entrypoint to run for each variant. Default: main.py",
    )
    parser.add_argument(
        "--python_bin",
        type=str,
        default=sys.executable,
        help="Python executable to use.",
    )
    parser.add_argument(
        "--set_model_name",
        action="store_true",
        help="Override the automatic experiment name with --model_name <dataset>_<variant-or-base>.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["readme"],
        help="Inject a predefined argument preset. readme uses the per-dataset hyperparameters from README.md.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Keep running remaining variants even if one command fails.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to the target script. Prefix them with --",
    )
    return parser.parse_args()


def normalize_extra_args(extra_args):
    if extra_args and extra_args[0] == "--":
        return extra_args[1:]
    return extra_args


def build_commands(args):
    commands = []
    for dataset in args.datasets:
        if dataset not in DATASET_META:
            raise ValueError(f"Unsupported dataset: {dataset}")

        variants = get_available_variants(dataset)
        if args.include_base:
            commands.append((dataset, None, build_command(args, dataset, None)))

        for variant in variants:
            commands.append((dataset, variant, build_command(args, dataset, variant)))

    return commands


def build_command(args, dataset, variant):
    command = [args.python_bin, args.script, "--data_name", dataset]
    if variant is not None:
        command.extend(["--data_variant", variant])
    if args.set_model_name:
        model_name = f"{dataset}_{variant or 'base'}"
        command.extend(["--model_name", model_name])
    if args.preset == "readme":
        command.extend(README_PRESET_ARGS[dataset])
    command.extend(normalize_extra_args(args.extra_args))
    return command


def main():
    args = parse_args()
    commands = build_commands(args)

    if not commands:
        raise ValueError("No datasets or variants matched the request.")

    for index, (dataset, variant, command) in enumerate(commands, start=1):
        label = f"{dataset}:{variant or 'base'}"
        print(f"[{index}/{len(commands)}] {label}")
        print(shlex.join(command))

        if args.dry_run:
            continue

        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            if not args.continue_on_error:
                raise SystemExit(completed.returncode)
            print(f"Command failed with exit code {completed.returncode}: {label}", file=sys.stderr)


if __name__ == "__main__":
    main()
