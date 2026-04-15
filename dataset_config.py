import json
import re
from pathlib import Path


DATASET_META = {
    "tmall": {
        "data_path": "./data/Tmall",
        "variant_root": "./data_variants/tmall",
        "behaviors": ["click", "collect", "cart", "buy"],
    },
    "jdata": {
        "data_path": "./data/jdata",
        "variant_root": "./data_variants/jdata",
        "behaviors": ["view", "collect", "cart", "buy"],
    },
    "taobao": {
        "data_path": "./data/taobao",
        "variant_root": "./data_variants/taobao",
        "behaviors": ["view", "cart", "buy"],
    },
}


def normalize_data_name(data_name):
    normalized = data_name.lower()
    if normalized not in DATASET_META:
        raise ValueError(f"Unsupported data_name: {data_name}")
    return normalized


def get_available_variants(data_name):
    data_name = normalize_data_name(data_name)
    variant_root = Path(DATASET_META[data_name]["variant_root"])
    if not variant_root.exists():
        return []
    return sorted(path.name for path in variant_root.iterdir() if path.is_dir())


def resolve_dataset_path(data_name, data_variant=None, data_path=None):
    data_name = normalize_data_name(data_name)
    if data_path:
        return str(Path(data_path))
    if data_variant:
        return str(Path(DATASET_META[data_name]["variant_root"]) / data_variant)
    return DATASET_META[data_name]["data_path"]


def infer_behaviors(data_name, data_path):
    data_name = normalize_data_name(data_name)
    dataset_dir = Path(data_path)
    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        behaviors = metadata.get("behaviors")
        if isinstance(behaviors, list) and behaviors:
            if behaviors[-1] != "buy":
                raise ValueError(
                    f"Dataset metadata at {metadata_path} must end with buy as the target behavior. "
                    f"Resolved behaviors: {behaviors}"
                )
            return behaviors

    behavior_order = DATASET_META[data_name]["behaviors"]
    behaviors = [behavior for behavior in behavior_order if (dataset_dir / f"{behavior}.txt").exists()]
    if not behaviors:
        raise FileNotFoundError(f"Could not infer behaviors from dataset path: {dataset_dir}")
    if behaviors[-1] != "buy":
        raise ValueError(
            f"Dataset at {dataset_dir} must contain buy.txt as the target behavior. "
            f"Resolved behaviors: {behaviors}"
        )
    return behaviors


def resolve_dataset_config(config):
    config = dict(config)
    data_name = normalize_data_name(config["data_name"])
    data_variant = config.get("data_variant") or None
    data_path = resolve_dataset_path(
        data_name=data_name,
        data_variant=data_variant,
        data_path=config.get("data_path"),
    )

    dataset_dir = Path(data_path)
    if not dataset_dir.exists():
        if data_variant:
            available_variants = get_available_variants(data_name)
            available_hint = (
                f" Available variants: {', '.join(available_variants)}"
                if available_variants else ""
            )
            raise FileNotFoundError(
                f"Dataset variant not found: {data_variant} ({dataset_dir}).{available_hint}"
            )
        raise FileNotFoundError(f"Dataset path not found: {dataset_dir}")

    config["data_name"] = data_name
    config["data_variant"] = data_variant
    config["data_path"] = str(dataset_dir)
    config["behaviors"] = infer_behaviors(data_name, dataset_dir)
    return config


def _slugify_token(value):
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return token or "custom"


def build_experiment_name(data_name, data_variant=None, data_path=None):
    data_name = normalize_data_name(data_name)
    if data_variant:
        return f"{data_name}_{_slugify_token(data_variant)}"

    default_path = Path(DATASET_META[data_name]["data_path"]).resolve()
    resolved_path = Path(data_path or DATASET_META[data_name]["data_path"]).resolve()
    if resolved_path == default_path:
        return data_name

    return f"{data_name}_{_slugify_token(resolved_path.name)}"
