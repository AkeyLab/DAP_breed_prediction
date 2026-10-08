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
    "model_path",
    "prediction_model_path",
    "model_metadata_path",
    "scaler_path",
    "pca_model_path",
)

_UNSET = object()


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


def run_mode(
    mode,
    config,
    *,
    base_dir=None,
    configure_logging=True,
    model_path=None,
    prediction_model_path=None,
    pretrained_model_name=None,
    prediction_model_type=None,
    scaler_path=None,
    pca_model_path=None,
    pure_threshold=_UNSET,
):
    """Run one pipeline mode from Python.

    Parameters
    ----------
    mode : int
        Pipeline mode from 1 through 5.
    config : Mapping or path-like
        Configuration dictionary or YAML file path.
    base_dir : path-like, optional
        Directory used to resolve relative paths in the configuration. Defaults
        to the current working directory, matching command-line behavior.
    configure_logging : bool, default=True
        Write ``process.log`` and emit pipeline logs to the console.
    model_path, prediction_model_path : path-like, optional
        Direct model path override. ``model_path`` is a convenience alias for
        ``prediction_model_path``. Relative paths are resolved against
        ``base_dir``. For Mode 1 custom PCA100 models, also provide
        ``pure_threshold`` and optionally ``prediction_model_type`` and
        ``scaler_path``.
    pretrained_model_name : str, optional
        Bundled Mode 1 model registry key, such as ``random_forest`` or
        ``xgboost``.
    prediction_model_type : str, optional
        Inference backend for a custom Mode 1 model. Defaults to ``sklearn``.
    scaler_path, pca_model_path : path-like, optional
        Optional direct preprocessing artifact overrides.
    pure_threshold : float, optional
        Direct pure-versus-mixed threshold override.

    Returns
    -------
    pathlib.Path
        The configured result directory.
    """
    try:
        mode = int(mode)
    except (TypeError, ValueError) as exc:
        raise ValueError("Mode must be an integer from 1 through 5.") from exc
    if mode not in range(1, 6):
        raise ValueError("Mode must be an integer from 1 through 5.")

    if model_path is not None and prediction_model_path is not None:
        raise ValueError("Use either model_path or prediction_model_path, not both.")

    loaded = load_config(config)
    if model_path is not None:
        loaded["model_path"] = model_path
    if prediction_model_path is not None:
        loaded["prediction_model_path"] = prediction_model_path
    if pretrained_model_name is not None:
        loaded["pretrained_model_name"] = pretrained_model_name
    if prediction_model_type is not None:
        loaded["prediction_model_type"] = prediction_model_type
    if scaler_path is not None:
        loaded["scaler_path"] = scaler_path
    if pca_model_path is not None:
        loaded["pca_model_path"] = pca_model_path
    if pure_threshold is not _UNSET:
        loaded["pure_threshold"] = pure_threshold

    base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    resolved = _resolve_config_paths(loaded, base_dir)
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
    include_unknown = resolved.get("include_unknown", True)
    pure_threshold = resolved.get("pure_threshold")
    pretrained_model_name = resolved.get("pretrained_model_name")
    prediction_model_type = resolved.get("prediction_model_type")
    if resolved.get("model_path") and resolved.get("prediction_model_path"):
        raise ValueError("Use either model_path or prediction_model_path, not both.")
    selected_model_path = resolved.get("prediction_model_path") or resolved.get("model_path")

    if mode == 1:
        _require(resolved, mode, "SNP_csv_path")
        pipeline.pretrained_inference(
            result_folder_path=str(result_path),
            SNP_csv_path=snp_csv_path,
            pca_model_path=resolved.get("pca_model_path"),
            prediction_model_path=selected_model_path,
            pretrained_model_name=pretrained_model_name,
            prediction_model_type=prediction_model_type,
            scaler_path=resolved.get("scaler_path"),
            pure_threshold=pure_threshold,
        )
    elif mode == 2:
        _require(resolved, mode, "SNP_csv_path")
        if pure_threshold is not None:
            logger.info(
                "Mode 2 ignores configured pure_threshold; the threshold is "
                "selected from training predictions."
            )
        training = pipeline.train(
            result_folder_path=str(result_path),
            SNP_csv_path=snp_csv_path,
            breed_list_text_path=breed_list_text_path,
            pca_components=pca_components,
            random_state=random_state,
            pure_threshold=None,
            include_unknown=include_unknown,
        )
        pipeline.inference(
            result_folder_path=str(result_path),
            SNP_csv_path=snp_csv_path,
            prediction_model_path=str(training["prediction_model_path"]),
            model_metadata_path=str(training["model_metadata_path"]),
            pure_threshold=training["pure_threshold"],
            require_exact_features=False,
        )
    elif mode == 3:
        if not selected_model_path:
            raise ValueError(
                "Mode 3 requires `prediction_model_path` or `model_path` in the configuration."
            )
        _require(
            resolved,
            mode,
            "SNP_csv_path",
            "pure_threshold",
        )
        pipeline.inference(
            result_folder_path=str(result_path),
            SNP_csv_path=snp_csv_path,
            prediction_model_path=selected_model_path,
            model_metadata_path=resolved.get("model_metadata_path"),
            scaler_path=resolved.get("scaler_path"),
            pca_model_path=resolved.get("pca_model_path"),
            pure_threshold=resolved["pure_threshold"],
            label_path=label_path,
            require_exact_features=True,
        )
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
    else:
        pipeline.full_training_pipeline(
            result_folder_path=str(result_path),
            breed_list_text_path="reproduce",
            pca_components=100,
            random_state=42,
            test_size=0.3,
        )

    return result_path
