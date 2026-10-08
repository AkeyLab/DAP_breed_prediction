"""Generate mode-specific YAML configs and matching CLI commands."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import yaml


MODE_DEFAULTS = {
    1: {
        "result_folder_path": "./results/mode_1",
        "SNP_csv_path": "./data/Toy_X_full_54143_single.csv",
        "pretrained_model_name": "random_forest",
        "prediction_model_path": None,
        "model_path": None,
        "prediction_model_type": None,
        "scaler_path": None,
        "pca_model_path": None,
        "pure_threshold": None,
        "configure_logging": True,
    },
    2: {
        "result_folder_path": "./results/mode_2",
        "SNP_csv_path": "./data/Toy_X_single.csv",
        "breed_list_text_path": "./data/paper_14_breed_list.txt",
        "include_unknown": False,
        "pca_components": 0.95,
        "random_state": 42,
        "pure_threshold": None,
        "configure_logging": True,
    },
    3: {
        "result_folder_path": "./results/mode_3",
        "prediction_model_path": "./results/mode_2/Model/Prediction_model_theta_0.62.pkl",
        "model_path": None,
        "model_metadata_path": "./results/mode_2/Model/model_metadata.json",
        "SNP_csv_path": "./data/Toy_X_single.csv",
        "pure_threshold": 0.62,
        "prediction_model_type": None,
        "scaler_path": None,
        "pca_model_path": None,
        "model_input_scaler_path": None,
        "label_path": None,
        "configure_logging": True,
    },
    4: {
        "result_folder_path": "./results/mode_4",
        "SNP_csv_path": "./data/Toy_X_snps.csv",
        "breed_list_text_path": "./data/Toy_a_short_breed_list.txt",
        "label_path": "./data/Toy_Y_labels.csv",
        "training_model_name": "random_forest",
        "xgboost_device": "auto",
        "pca_components": 0.95,
        "random_state": 42,
        "test_size": 0.3,
        "configure_logging": True,
    },
    5: {
        "result_folder_path": "./results/mode_5",
        "random_state": 42,
        "pca_components": 100,
        "test_size": 0.3,
        "configure_logging": True,
    },
}

MODE_KEYS = {mode: tuple(defaults) for mode, defaults in MODE_DEFAULTS.items()}
MODEL_CHOICES = (
    "random_forest",
    "xgboost",
    "ridge",
    "knn",
    "extratrees",
    "mlp",
    "transformer",
)


class NullWhenEmpty(argparse.Action):
    """Convert empty strings to None so users can clear optional defaults."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, None if values == "" else values)


def _add_common_arguments(parser):
    parser.add_argument(
        "--mode",
        type=int,
        choices=range(1, 6),
        required=True,
        metavar="{1,2,3,4,5}",
        help="Pipeline mode to configure.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="YAML file to write. Defaults to configs/generated_mode_<mode>.yml.",
    )
    parser.add_argument(
        "--python",
        default="python",
        help="Python executable used in the generated run command.",
    )
    parser.add_argument(
        "--entry-script",
        default="main.py",
        help="Entry script used in the generated run command.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite config_path if it already exists.",
    )


def _add_config_arguments(parser):
    parser.add_argument(
        "--result-folder-path", dest="result_folder_path", default=argparse.SUPPRESS
    )
    parser.add_argument("--snp-csv-path", dest="SNP_csv_path", default=argparse.SUPPRESS)
    parser.add_argument("--label-path", action=NullWhenEmpty, default=argparse.SUPPRESS)
    parser.add_argument(
        "--breed-list-text-path", action=NullWhenEmpty, default=argparse.SUPPRESS
    )
    parser.add_argument("--model-path", action=NullWhenEmpty, default=argparse.SUPPRESS)
    parser.add_argument(
        "--prediction-model-path", action=NullWhenEmpty, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--model-metadata-path", action=NullWhenEmpty, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--pretrained-model-name", choices=MODEL_CHOICES, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--prediction-model-type",
        choices=("sklearn", "torch_mlp", "torch_transformer"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--scaler-path", action=NullWhenEmpty, default=argparse.SUPPRESS)
    parser.add_argument("--pca-model-path", action=NullWhenEmpty, default=argparse.SUPPRESS)
    parser.add_argument(
        "--model-input-scaler-path", action=NullWhenEmpty, default=argparse.SUPPRESS
    )
    parser.add_argument("--pure-threshold", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--pca-components", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--random-state", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--test-size", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--include-unknown",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Mode 2 class-subset behavior.",
    )
    parser.add_argument(
        "--training-model-name", choices=MODEL_CHOICES, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--xgboost-device",
        choices=("auto", "cpu", "cuda"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--configure-logging",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Write process.log and console logs.",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a DAP breed-prediction YAML config and the matching "
            "`python main.py -mode ... -config_path ...` command."
        )
    )
    _add_common_arguments(parser)
    _add_config_arguments(parser)
    return parser


def _selected_overrides(args):
    ignored = {"mode", "config_path", "python", "entry_script", "force"}
    overrides = {}
    for key, value in vars(args).items():
        if key in ignored:
            continue
        overrides[key] = value
    return overrides


def _validate_overrides(mode, overrides):
    allowed = set(MODE_KEYS[mode])
    unsupported = sorted(set(overrides) - allowed)
    if unsupported:
        names = ", ".join(f"--{name.replace('_', '-')}" for name in unsupported)
        raise ValueError(f"Mode {mode} does not use: {names}")
    if overrides.get("model_path") and overrides.get("prediction_model_path"):
        raise ValueError("Use either --model-path or --prediction-model-path, not both.")


def build_config(mode, overrides=None):
    """Return a mode-specific config dictionary with overrides applied."""
    if mode not in MODE_DEFAULTS:
        raise ValueError("mode must be an integer from 1 through 5.")
    overrides = dict(overrides or {})
    _validate_overrides(mode, overrides)
    config = dict(MODE_DEFAULTS[mode])
    for key, value in overrides.items():
        config[key] = value
    if config.get("model_path"):
        config["prediction_model_path"] = None
    if config.get("prediction_model_path"):
        config["model_path"] = None
    return config


def default_config_path(mode):
    return Path("configs") / f"generated_mode_{mode}.yml"


def write_config(config, config_path, force=False):
    path = Path(config_path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Re-run with --force to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def build_run_command(mode, config_path, python="python", entry_script="main.py"):
    parts = [
        python,
        entry_script,
        "-mode",
        str(mode),
        "-config_path",
        str(config_path),
    ]
    return " ".join(shlex.quote(part) for part in parts)


def generate(args):
    overrides = _selected_overrides(args)
    config = build_config(args.mode, overrides)
    config_path = Path(args.config_path) if args.config_path else default_config_path(args.mode)
    written_path = write_config(config, config_path, force=args.force)
    command = build_run_command(args.mode, written_path, args.python, args.entry_script)
    return written_path, command


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        path, command = generate(args)
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote YAML config: {path}")
    print("Run this command:")
    print(command)


if __name__ == "__main__":
    main()
