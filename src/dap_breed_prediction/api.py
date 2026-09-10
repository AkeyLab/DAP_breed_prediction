"""Public Python API for running DAP breed-prediction workflows."""

import logging
from collections.abc import Mapping
from pathlib import Path

import yaml

from . import pipeline


PATH_KEYS = (
    "result_folder_path",
    "SNP_csv_path",
    "label_path",
    "breed_list_text_path",
)


def load_config(config):
    """Return a mutable config dictionary from a mapping or YAML path."""
    if isinstance(config, Mapping):
        return dict(config)

    config_path = Path(config)
    with config_path.open() as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError("The YAML configuration must contain a key-value mapping.")
    return dict(loaded)


def setup_logger(result_folder_path):
    """Configure console and file logging for a pipeline run."""
    log_path = Path(result_folder_path) / "process.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def _resolve_config_paths(config, base_dir):
    resolved = dict(config)
    base_dir = Path(base_dir).expanduser().resolve()
    for key in PATH_KEYS:
        value = resolved.get(key)
        if value is None:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        resolved[key] = str(path.resolve())
    return resolved


def _require(config, mode, *keys):
    missing = [key for key in keys if not config.get(key)]
    if missing:
        joined = ", ".join(f"`{key}`" for key in missing)
        raise ValueError(f"Mode {mode} requires {joined} in the configuration.")


def run_mode(mode, config, *, base_dir=None, configure_logging=True):
    """Run one pipeline mode from Python.

    Parameters
    ----------
    mode : int
        Pipeline mode from 1 through 6.
    config : Mapping or path-like
        Configuration dictionary or YAML file path.
    base_dir : path-like, optional
        Directory used to resolve relative paths in the configuration. Defaults
        to the current working directory, matching command-line behavior.
    configure_logging : bool, default=True
        Write ``process.log`` and emit pipeline logs to the console.

    Returns
    -------
    pathlib.Path
        The configured result directory.
    """
    try:
        mode = int(mode)
    except (TypeError, ValueError) as exc:
        raise ValueError("Mode must be an integer from 1 through 6.") from exc
    if mode not in range(1, 7):
        raise ValueError("Mode must be an integer from 1 through 6.")

    base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    resolved = _resolve_config_paths(load_config(config), base_dir)
    _require(resolved, mode, "result_folder_path")

    result_path = Path(resolved["result_folder_path"])
    logger = setup_logger(result_path) if configure_logging else logging.getLogger(__name__)
    logger.info("Running DAP breed-prediction Mode %s", mode)

    snp_csv_path = resolved.get("SNP_csv_path")
    label_path = resolved.get("label_path")
    breed_list_text_path = resolved.get("breed_list_text_path")
    pca_components = resolved.get("pca_components", 0.95)
    random_state = resolved.get("random_state", 42)
    test_size = resolved.get("test_size", 0.3)

    if mode == 1:
        _require(resolved, mode, "SNP_csv_path")
        input_args = {
            "result_folder_path": str(result_path),
            "SNP_csv_path": snp_csv_path,
            "pca_components": pca_components,
            "random_state": random_state,
        }
        pipeline.train(**input_args)
        pipeline.inference(**input_args)
    elif mode == 2:
        _require(resolved, mode, "SNP_csv_path", "breed_list_text_path")
        input_args = {
            "result_folder_path": str(result_path),
            "SNP_csv_path": snp_csv_path,
            "breed_list_text_path": breed_list_text_path,
            "pca_components": pca_components,
            "random_state": random_state,
        }
        pipeline.train(**input_args)
        pipeline.inference(**input_args)
    elif mode == 3:
        _require(resolved, mode, "SNP_csv_path", "label_path")
        input_args = {
            "result_folder_path": str(result_path),
            "SNP_csv_path": snp_csv_path,
            "label_path": label_path,
            "breed_list_text_path": breed_list_text_path,
            "pca_components": pca_components,
            "random_state": random_state,
        }
        pipeline.train(**input_args)
        pipeline.inference(**input_args)
    elif mode == 4:
        _require(resolved, mode, "SNP_csv_path", "label_path")
        pipeline.full_training_pipeline(
            result_folder_path=str(result_path),
            SNP_csv_path=snp_csv_path,
            label_path=label_path,
            breed_list_text_path=breed_list_text_path,
            pca_components=pca_components,
            random_state=random_state,
            test_size=test_size,
        )
    elif mode == 5:
        _require(resolved, mode, "breed_list_text_path")
        pipeline.full_training_pipeline(
            result_folder_path=str(result_path),
            breed_list_text_path=breed_list_text_path,
            pca_components=pca_components,
            random_state=random_state,
            test_size=test_size,
        )
    else:
        pipeline.full_training_pipeline(
            result_folder_path=str(result_path),
            breed_list_text_path="reproduce",
            pca_components=100,
            random_state=42,
            test_size=0.3,
        )

    return result_path
