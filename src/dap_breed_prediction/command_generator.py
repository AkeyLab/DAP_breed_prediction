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
        metavar="{1,2,3,4,5}",
        help="Pipeline mode to configure. Required unless --interactive is used.",
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
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask a short decision tree and then write the matching YAML config.",
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
    ignored = {"mode", "config_path", "python", "entry_script", "force", "interactive"}
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


def _prompt_text(message, default=None, input_func=input, output_func=print):
    suffix = f" [{default}]" if default not in (None, "") else ""
    output_func(f"{message}{suffix}")
    answer = input_func("> ").strip()
    if answer == "":
        return default
    if answer.lower() in {"none", "null"}:
        return None
    return answer


def _prompt_bool(message, default=False, input_func=input, output_func=print):
    default_text = "Y/n" if default else "y/N"
    while True:
        output_func(f"{message} [{default_text}]")
        answer = input_func("> ").strip().lower()
        if answer == "":
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        output_func("Please answer yes or no.")


def _prompt_choice(message, choices, default=None, input_func=input, output_func=print):
    choices = list(choices)
    default = default if default is not None else choices[0]
    while True:
        output_func(message)
        for index, choice in enumerate(choices, start=1):
            marker = " (default)" if choice == default else ""
            output_func(f"  {index}. {choice}{marker}")
        answer = input_func("> ").strip()
        if answer == "":
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        if answer in choices:
            return answer
        output_func("Please enter one of the listed numbers or values.")


def _prompt_number(message, default, cast, input_func=input, output_func=print):
    while True:
        answer = _prompt_text(message, default, input_func, output_func)
        try:
            return cast(answer)
        except (TypeError, ValueError):
            output_func(f"Please enter a valid {cast.__name__}.")


def select_mode_interactively(input_func=input, output_func=print):
    """Return the pipeline mode selected by the decision tree."""
    if _prompt_bool(
        "Do you want to reproduce the fixed paper benchmark?",
        default=False,
        input_func=input_func,
        output_func=output_func,
    ):
        return 5
    if _prompt_bool(
        "Do you need to train or retrain a model?",
        default=False,
        input_func=input_func,
        output_func=output_func,
    ):
        if _prompt_bool(
            "Is the training data your own labeled X/Y dataset, without DAP reference genotypes?",
            default=True,
            input_func=input_func,
            output_func=output_func,
        ):
            return 4
        return 2
    if _prompt_bool(
        "Are you using a bundled pretrained 100-class model on one full 54,143-SNP sample?",
        default=True,
        input_func=input_func,
        output_func=output_func,
    ):
        return 1
    return 3


def _configure_common(config, input_func, output_func):
    config["result_folder_path"] = _prompt_text(
        "Result folder path",
        config["result_folder_path"],
        input_func,
        output_func,
    )
    config["configure_logging"] = _prompt_bool(
        "Write process.log and console logs?",
        default=bool(config.get("configure_logging", True)),
        input_func=input_func,
        output_func=output_func,
    )


def _configure_mode_1(config, input_func, output_func):
    config["SNP_csv_path"] = _prompt_text(
        "Input SNP CSV path",
        config["SNP_csv_path"],
        input_func,
        output_func,
    )
    if _prompt_bool(
        "Use a custom PCA100 prediction model instead of the bundled registry?",
        default=False,
        input_func=input_func,
        output_func=output_func,
    ):
        config["pretrained_model_name"] = None
        config["prediction_model_path"] = _prompt_text(
            "Custom prediction model path", None, input_func, output_func
        )
        config["prediction_model_type"] = _prompt_choice(
            "Model type",
            ("sklearn", "torch_mlp", "torch_transformer"),
            default="sklearn",
            input_func=input_func,
            output_func=output_func,
        )
        config["scaler_path"] = _prompt_text(
            "Optional model-input scaler path (blank for none)",
            None,
            input_func,
            output_func,
        )
        config["pca_model_path"] = _prompt_text(
            "PCA model path",
            "./model/pca_model_WG_100.joblib",
            input_func,
            output_func,
        )
        config["pure_threshold"] = _prompt_number(
            "Pure-versus-mixed threshold",
            0.7,
            float,
            input_func,
            output_func,
        )
    else:
        config["pretrained_model_name"] = _prompt_choice(
            "Bundled pretrained model",
            MODEL_CHOICES,
            default=config["pretrained_model_name"],
            input_func=input_func,
            output_func=output_func,
        )
        config["pure_threshold"] = _prompt_text(
            "Optional threshold override (blank uses model default)",
            None,
            input_func,
            output_func,
        )
        if config["pure_threshold"] is not None:
            config["pure_threshold"] = float(config["pure_threshold"])


def _configure_mode_2(config, input_func, output_func):
    config["SNP_csv_path"] = _prompt_text(
        "Input SNP CSV path",
        config["SNP_csv_path"],
        input_func,
        output_func,
    )
    if _prompt_bool(
        "Restrict outputs to a breed-list file?",
        default=True,
        input_func=input_func,
        output_func=output_func,
    ):
        config["breed_list_text_path"] = _prompt_text(
            "Breed-list text path",
            config["breed_list_text_path"],
            input_func,
            output_func,
        )
        config["include_unknown"] = _prompt_bool(
            "Include the Unknown class?",
            default=bool(config.get("include_unknown", False)),
            input_func=input_func,
            output_func=output_func,
        )
    else:
        config["breed_list_text_path"] = None
        config["include_unknown"] = True
    config["pca_components"] = _prompt_number(
        "PCA components or variance fraction",
        config["pca_components"],
        float,
        input_func,
        output_func,
    )
    config["random_state"] = _prompt_number(
        "Random seed", config["random_state"], int, input_func, output_func
    )


def _configure_mode_3(config, input_func, output_func):
    config["SNP_csv_path"] = _prompt_text(
        "Input SNP CSV path",
        config["SNP_csv_path"],
        input_func,
        output_func,
    )
    config["prediction_model_path"] = _prompt_text(
        "Prediction model path",
        config["prediction_model_path"],
        input_func,
        output_func,
    )
    config["model_path"] = None
    config["model_metadata_path"] = _prompt_text(
        "Model metadata path",
        config["model_metadata_path"],
        input_func,
        output_func,
    )
    config["pure_threshold"] = _prompt_number(
        "Pure-versus-mixed threshold",
        config["pure_threshold"],
        float,
        input_func,
        output_func,
    )
    if _prompt_bool(
        "Do you have labels for performance analysis?",
        default=False,
        input_func=input_func,
        output_func=output_func,
    ):
        config["label_path"] = _prompt_text("Label CSV path", None, input_func, output_func)
    if _prompt_bool(
        "Override scaler/PCA sidecar paths from metadata?",
        default=False,
        input_func=input_func,
        output_func=output_func,
    ):
        config["prediction_model_type"] = _prompt_text(
            "Optional model type override", None, input_func, output_func
        )
        config["scaler_path"] = _prompt_text(
            "Optional raw-feature scaler path", None, input_func, output_func
        )
        config["pca_model_path"] = _prompt_text(
            "Optional PCA model path", None, input_func, output_func
        )
        config["model_input_scaler_path"] = _prompt_text(
            "Optional model-input scaler path", None, input_func, output_func
        )


def _configure_mode_4(config, input_func, output_func):
    config["SNP_csv_path"] = _prompt_text(
        "Training SNP CSV path", config["SNP_csv_path"], input_func, output_func
    )
    config["label_path"] = _prompt_text(
        "Training label CSV path", config["label_path"], input_func, output_func
    )
    if _prompt_bool(
        "Use a breed-list file to define output classes?",
        default=True,
        input_func=input_func,
        output_func=output_func,
    ):
        config["breed_list_text_path"] = _prompt_text(
            "Breed-list text path",
            config["breed_list_text_path"],
            input_func,
            output_func,
        )
    else:
        config["breed_list_text_path"] = None
    config["training_model_name"] = _prompt_choice(
        "Training model",
        MODEL_CHOICES,
        default=config["training_model_name"],
        input_func=input_func,
        output_func=output_func,
    )
    if config["training_model_name"] == "xgboost":
        config["xgboost_device"] = _prompt_choice(
            "XGBoost device",
            ("auto", "cpu", "cuda"),
            default=config["xgboost_device"],
            input_func=input_func,
            output_func=output_func,
        )
    config["pca_components"] = _prompt_number(
        "PCA components or variance fraction",
        config["pca_components"],
        float,
        input_func,
        output_func,
    )
    config["random_state"] = _prompt_number(
        "Random seed", config["random_state"], int, input_func, output_func
    )
    config["test_size"] = _prompt_number(
        "Held-out test fraction", config["test_size"], float, input_func, output_func
    )


def _configure_mode_5(config, input_func, output_func):
    if _prompt_bool(
        "Customize benchmark defaults?",
        default=False,
        input_func=input_func,
        output_func=output_func,
    ):
        config["random_state"] = _prompt_number(
            "Random seed", config["random_state"], int, input_func, output_func
        )
        config["pca_components"] = _prompt_number(
            "PCA components",
            config["pca_components"],
            int,
            input_func,
            output_func,
        )
        config["test_size"] = _prompt_number(
            "Held-out test fraction",
            config["test_size"],
            float,
            input_func,
            output_func,
        )


MODE_CONFIGURATORS = {
    1: _configure_mode_1,
    2: _configure_mode_2,
    3: _configure_mode_3,
    4: _configure_mode_4,
    5: _configure_mode_5,
}


def interactive_config(input_func=input, output_func=print, ask_config_path=True):
    """Ask the decision tree and return ``(mode, config, config_path)``."""
    output_func("DAP breed-prediction command generator")
    mode = select_mode_interactively(input_func=input_func, output_func=output_func)
    output_func(f"Selected Mode {mode}.")
    config = build_config(mode)
    _configure_common(config, input_func, output_func)
    MODE_CONFIGURATORS[mode](config, input_func, output_func)
    config_path = None
    if ask_config_path:
        config_path = _prompt_text(
            "YAML config output path",
            str(default_config_path(mode)),
            input_func,
            output_func,
        )
    return mode, config, config_path


def generate(args):
    if args.interactive:
        mode, config, prompted_config_path = interactive_config(
            ask_config_path=args.config_path is None
        )
        config_path = Path(args.config_path or prompted_config_path)
    else:
        if args.mode is None:
            raise ValueError("--mode is required unless --interactive is used.")
        mode = args.mode
        overrides = _selected_overrides(args)
        config = build_config(mode, overrides)
        config_path = Path(args.config_path) if args.config_path else default_config_path(mode)
    written_path = write_config(config, config_path, force=args.force)
    command = build_run_command(mode, written_path, args.python, args.entry_script)
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
