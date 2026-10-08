#!/usr/bin/env python
"""Train the bundled PCA100 model set used by Mode 1.

The script trains each configured model on the split-specific PCA100 training
matrix in ``data/`` and evaluates on the matching held-out test matrix. These
PCs were produced by fitting scaling and PCA on training dogs only.

Examples
--------
Train all supported models into a scratch folder:

    python scripts/train_pretrained_models.py --output-dir /tmp/dap_models

Train only the CPU baseline models:

    python scripts/train_pretrained_models.py --models random_forest,ridge,knn,extratrees

The saved artifacts are suitable for adding to ``model/model_registry.json`` or
using directly with ``run_mode(..., model_path=...)``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "model" / "trained_models"
X_TRAIN_PATH = DATA_DIR / "X_train_SNP_WG_prune_v3_1_std_pca_100.csv"
X_TEST_PATH = DATA_DIR / "X_test_SNP_WG_prune_v3_1_std_pca_100.csv"
Y_PATH = DATA_DIR / "y_combined_100.csv"


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def gpu_names() -> list[str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def hardware_summary() -> dict[str, object]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        "gpus": gpu_names(),
    }


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_train = pd.read_csv(X_TRAIN_PATH, index_col="dog_id")
    x_test = pd.read_csv(X_TEST_PATH, index_col="dog_id")
    y_all = pd.read_csv(Y_PATH, index_col="dog_id")
    x_train.index = x_train.index.astype(str)
    x_test.index = x_test.index.astype(str)
    y_all.index = y_all.index.astype(str)
    y_train = y_all.loc[x_train.index]
    y_test = y_all.loc[x_test.index]
    return x_train, x_test, y_train, y_test


def transform_prediction(y_pred: np.ndarray, pure_threshold: float) -> np.ndarray:
    transformed = []
    for row in y_pred:
        if row.max() > pure_threshold:
            one_hot = np.zeros_like(row, dtype=float)
            one_hot[np.argmax(row)] = 1.0
            transformed.append(one_hot)
        else:
            top2 = np.argsort(row)[-2:]
            mixed = np.zeros_like(row, dtype=float)
            mixed[top2] = 0.5
            transformed.append(mixed)
    return np.array(transformed)


def prediction_analysis(y_pred: np.ndarray, y_true: pd.DataFrame, threshold: float) -> tuple[float, float, float]:
    y_transformed = transform_prediction(y_pred, threshold)
    y_true_arr = y_true.to_numpy()
    strict = np.all(y_transformed == y_true_arr, axis=1)
    loose = []
    for true_row, pred_row in zip(y_true_arr, y_transformed):
        true_indices = set(np.where(true_row > 0)[0])
        pred_indices = set(np.where(pred_row > 0)[0])
        loose.append(len(true_indices & pred_indices) > 0)
    true_is_pure = y_true_arr.max(axis=1) == 1.0
    pred_is_pure = y_transformed.max(axis=1) == 1.0
    return float(np.mean(strict)), float(np.mean(loose)), float(np.mean(true_is_pure == pred_is_pure))


def threshold_search(y_pred: np.ndarray, y_true: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0, 1.01, 0.01):
        strict, loose, pure_mix = prediction_analysis(y_pred, y_true, float(threshold))
        rows.append(
            {
                "pure_threshold": float(threshold),
                "strict_accuracy": strict,
                "loose_accuracy": loose,
                "pure_mix_accuracy": pure_mix,
            }
        )
    return pd.DataFrame(rows)


def best_pure_threshold(threshold_df: pd.DataFrame) -> float:
    max_loose = threshold_df["loose_accuracy"].max()
    candidates = threshold_df[threshold_df["loose_accuracy"].eq(max_loose)]
    max_strict = candidates["strict_accuracy"].max()
    candidates = candidates[candidates["strict_accuracy"].eq(max_strict)]
    return float(candidates["pure_threshold"].iloc[-1])


def evaluate(
    name: str,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray,
    train_seconds: float,
    train_infer_seconds: float,
    test_infer_seconds: float,
    notes: str,
    threshold_dir: Path,
) -> dict[str, object]:
    threshold_df = threshold_search(y_train_pred, y_train)
    threshold = best_pure_threshold(threshold_df)
    threshold_df.to_csv(threshold_dir / f"{name}_threshold_search_train.csv", index=False)
    strict, loose, pure_mix = prediction_analysis(y_test_pred, y_test, threshold)
    return {
        "model": name,
        "pure_threshold": threshold,
        "strict_accuracy": strict,
        "loose_accuracy": loose,
        "pure_mix_accuracy": pure_mix,
        "train_mse": float(mean_squared_error(y_train, y_train_pred)),
        "test_mse": float(mean_squared_error(y_test, y_test_pred)),
        "train_seconds": train_seconds,
        "train_inference_seconds": train_infer_seconds,
        "test_inference_seconds": test_infer_seconds,
        "inference_seconds": train_infer_seconds + test_infer_seconds,
        "total_seconds": train_seconds + train_infer_seconds + test_infer_seconds,
        "notes": notes,
    }


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_type: str
    artifact_name: str
    scaler_artifact_name: str | None
    trainer: Callable
    notes: str


def fit_sklearn_model(
    name: str,
    model,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    output_dir: Path,
    threshold_dir: Path,
    artifact_name: str,
    scaler_artifact_name: str | None = None,
    scale_features: bool = False,
    compress: int = 3,
    notes: str = "",
) -> dict[str, object]:
    train_start = time.time()
    if scale_features:
        scaler = StandardScaler()
        x_train_fit = scaler.fit_transform(x_train)
        joblib.dump(scaler, output_dir / scaler_artifact_name, compress=compress)
    else:
        x_train_fit = x_train
    model.fit(x_train_fit, y_train)
    train_seconds = time.time() - train_start
    joblib.dump(model, output_dir / artifact_name, compress=compress)

    train_pred_start = time.time()
    y_train_pred = np.asarray(model.predict(x_train_fit))
    train_infer_seconds = time.time() - train_pred_start

    test_pred_start = time.time()
    if scale_features:
        x_test_fit = scaler.transform(x_test)
    else:
        x_test_fit = x_test
    y_test_pred = np.asarray(model.predict(x_test_fit))
    test_infer_seconds = time.time() - test_pred_start

    result = evaluate(
        name,
        y_train,
        y_test,
        y_train_pred,
        y_test_pred,
        train_seconds,
        train_infer_seconds,
        test_infer_seconds,
        notes,
        threshold_dir,
    )
    result.update(
        {
            "model_type": "sklearn",
            "model_path": artifact_name,
            "scaler_path": scaler_artifact_name if scale_features else None,
        }
    )
    return result


def train_random_forest(*, x_train, x_test, y_train, y_test, output_dir, threshold_dir, seed, compress, **_):
    return fit_sklearn_model(
        "random_forest",
        MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1),
            n_jobs=-1,
        ),
        x_train,
        x_test,
        y_train,
        y_test,
        output_dir,
        threshold_dir,
        "random_forest-PCA100.joblib",
        compress=compress,
        notes="MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1), n_jobs=-1)",
    )


def train_xgboost(*, x_train, x_test, y_train, y_test, output_dir, threshold_dir, seed, compress, xgboost_device, **_):
    try:
        import xgboost as xgb
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install xgboost to train the XGBoost model.") from exc
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        multi_strategy="one_output_per_tree",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        device=xgboost_device,
        random_state=seed,
        n_jobs=-1,
    )
    return fit_sklearn_model(
        "xgboost",
        model,
        x_train,
        x_test,
        y_train,
        y_test,
        output_dir,
        threshold_dir,
        "xgboost-PCA100.joblib",
        compress=compress,
        notes=f"XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.85, colsample_bytree=0.85, tree_method='hist', device={xgboost_device!r})",
    )


def train_ridge(*, x_train, x_test, y_train, y_test, output_dir, threshold_dir, compress, **_):
    return fit_sklearn_model(
        "ridge",
        Ridge(alpha=10.0),
        x_train,
        x_test,
        y_train,
        y_test,
        output_dir,
        threshold_dir,
        "ridge-PCA100.joblib",
        scaler_artifact_name="ridge_scaler-PCA100.joblib",
        scale_features=True,
        compress=compress,
        notes="StandardScaler() followed by Ridge(alpha=10.0)",
    )


def train_knn(*, x_train, x_test, y_train, y_test, output_dir, threshold_dir, compress, **_):
    return fit_sklearn_model(
        "knn",
        KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1),
        x_train,
        x_test,
        y_train,
        y_test,
        output_dir,
        threshold_dir,
        "knn-PCA100.joblib",
        scaler_artifact_name="knn_scaler-PCA100.joblib",
        scale_features=True,
        compress=compress,
        notes="StandardScaler() followed by KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=-1)",
    )


def train_extratrees(*, x_train, x_test, y_train, y_test, output_dir, threshold_dir, seed, compress, **_):
    return fit_sklearn_model(
        "extratrees",
        ExtraTreesRegressor(
            n_estimators=300,
            max_features=0.75,
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1,
        ),
        x_train,
        x_test,
        y_train,
        y_test,
        output_dir,
        threshold_dir,
        "extratrees-PCA100.joblib",
        compress=compress,
        notes="ExtraTreesRegressor(n_estimators=300, max_features=0.75, min_samples_leaf=1, random_state=seed, n_jobs=-1)",
    )


def make_mlp(torch, n_features: int, n_outputs: int):
    return torch.nn.Sequential(
        torch.nn.Linear(n_features, 256),
        torch.nn.LayerNorm(256),
        torch.nn.GELU(),
        torch.nn.Dropout(0.20),
        torch.nn.Linear(256, 128),
        torch.nn.LayerNorm(128),
        torch.nn.GELU(),
        torch.nn.Dropout(0.15),
        torch.nn.Linear(128, n_outputs),
    )


def make_transformer(torch, n_features: int, n_outputs: int):
    class PCTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            d_model = 64
            self.value_embed = torch.nn.Linear(1, d_model)
            self.pos_embed = torch.nn.Parameter(torch.zeros(1, n_features, d_model))
            layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=128,
                dropout=0.15,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(layer, num_layers=2)
            self.head = torch.nn.Sequential(
                torch.nn.LayerNorm(d_model),
                torch.nn.Linear(d_model, 128),
                torch.nn.GELU(),
                torch.nn.Dropout(0.10),
                torch.nn.Linear(128, n_outputs),
            )

        def forward(self, x):
            x = x.unsqueeze(-1)
            x = self.value_embed(x) + self.pos_embed
            x = self.encoder(x)
            x = x.mean(dim=1)
            return self.head(x)

    return PCTransformer()


def train_torch_classifier(
    name: str,
    model_factory: Callable,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    output_dir: Path,
    threshold_dir: Path,
    artifact_name: str,
    scaler_artifact_name: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    notes: str,
) -> dict[str, object]:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install torch to train PyTorch models.") from exc

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False

    train_start = time.time()
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype("float32")
    y_train_arr = y_train.to_numpy(dtype="float32")
    train_idx, val_idx = train_test_split(
        np.arange(len(x_train_scaled)), test_size=0.15, random_state=seed
    )
    train_ds = TensorDataset(
        torch.from_numpy(x_train_scaled[train_idx]),
        torch.from_numpy(y_train_arr[train_idx]),
    )
    val_x = torch.from_numpy(x_train_scaled[val_idx]).to(device)
    val_y = torch.from_numpy(y_train_arr[val_idx]).to(device)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model = model_factory(torch, x_train.shape[1], y_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def soft_target_ce(logits, target):
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        return -(target * log_probs).sum(dim=1).mean()

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = soft_target_ce(model(xb), yb)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(soft_target_ce(model(val_x), val_y).detach().cpu())
        rows.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    joblib.dump(scaler, output_dir / scaler_artifact_name, compress=3)
    torch.save(
        {
            "model_name": name,
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
        },
        output_dir / artifact_name,
    )
    pd.DataFrame(rows).to_csv(threshold_dir / f"{name}_training_history.csv", index=False)
    train_seconds = time.time() - train_start

    def predict(arr: np.ndarray) -> np.ndarray:
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(arr), 1024):
                xb = torch.from_numpy(arr[start:start + 1024]).to(device)
                probs = torch.softmax(model(xb), dim=1)
                preds.append(probs.detach().cpu().numpy())
        return np.vstack(preds)

    train_pred_start = time.time()
    y_train_pred = predict(x_train_scaled)
    train_infer_seconds = time.time() - train_pred_start
    test_pred_start = time.time()
    y_test_pred = predict(scaler.transform(x_test).astype("float32"))
    test_infer_seconds = time.time() - test_pred_start
    result = evaluate(
        name,
        y_train,
        y_test,
        y_train_pred,
        y_test_pred,
        train_seconds,
        train_infer_seconds,
        test_infer_seconds,
        notes,
        threshold_dir,
    )
    result.update(
        {
            "model_type": f"torch_{name}",
            "model_path": artifact_name,
            "scaler_path": scaler_artifact_name,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
        }
    )
    return result


def train_mlp(*, x_train, x_test, y_train, y_test, output_dir, threshold_dir, seed, **_):
    return train_torch_classifier(
        "mlp",
        make_mlp,
        x_train,
        x_test,
        y_train,
        y_test,
        output_dir,
        threshold_dir,
        "mlp-PCA100.pt",
        "mlp_scaler-PCA100.joblib",
        seed=seed,
        epochs=220,
        batch_size=256,
        lr=1e-3,
        patience=30,
        notes="PyTorch MLP: Linear(100,256)-LayerNorm-GELU-Dropout(0.20)-Linear(256,128)-LayerNorm-GELU-Dropout(0.15)-Linear(128,100); AdamW(lr=1e-3, weight_decay=1e-4)",
    )


def train_transformer(*, x_train, x_test, y_train, y_test, output_dir, threshold_dir, seed, **_):
    return train_torch_classifier(
        "transformer",
        make_transformer,
        x_train,
        x_test,
        y_train,
        y_test,
        output_dir,
        threshold_dir,
        "transformer-PCA100.pt",
        "transformer_scaler-PCA100.joblib",
        seed=seed,
        epochs=220,
        batch_size=256,
        lr=8e-4,
        patience=30,
        notes="Two-layer TransformerEncoder over 100 PC tokens: d_model=64, nhead=4, dim_feedforward=128, dropout=0.15, GELU; AdamW(lr=8e-4, weight_decay=1e-4)",
    )


MODEL_CONFIGS = {
    "random_forest": ModelConfig("random_forest", "sklearn", "random_forest-PCA100.joblib", None, train_random_forest, "CPU random forest baseline"),
    "xgboost": ModelConfig("xgboost", "sklearn", "xgboost-PCA100.joblib", None, train_xgboost, "XGBoost regressor"),
    "ridge": ModelConfig("ridge", "sklearn", "ridge-PCA100.joblib", "ridge_scaler-PCA100.joblib", train_ridge, "Ridge regression"),
    "knn": ModelConfig("knn", "sklearn", "knn-PCA100.joblib", "knn_scaler-PCA100.joblib", train_knn, "KNN regression"),
    "extratrees": ModelConfig("extratrees", "sklearn", "extratrees-PCA100.joblib", None, train_extratrees, "ExtraTrees regressor"),
    "mlp": ModelConfig("mlp", "torch_mlp", "mlp-PCA100.pt", "mlp_scaler-PCA100.joblib", train_mlp, "PyTorch MLP"),
    "transformer": ModelConfig("transformer", "torch_transformer", "transformer-PCA100.pt", "transformer_scaler-PCA100.joblib", train_transformer, "PyTorch PC-token Transformer"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model keys or 'all'. Available: " + ", ".join(MODEL_CONFIGS),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compress", type=int, default=3, help="Joblib compression level.")
    parser.add_argument(
        "--xgboost-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="XGBoost device. 'auto' uses cuda when nvidia-smi is available, else cpu.",
    )
    return parser.parse_args()


def selected_model_keys(selection: str) -> list[str]:
    if selection == "all":
        return list(MODEL_CONFIGS)
    keys = [item.strip() for item in selection.split(",") if item.strip()]
    unknown = sorted(set(keys) - set(MODEL_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown model key(s): {unknown}. Available: {sorted(MODEL_CONFIGS)}")
    return keys


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    threshold_dir = output_dir / "thresholds"
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_dir.mkdir(parents=True, exist_ok=True)
    x_train, x_test, y_train, y_test = load_data()
    env = hardware_summary()
    (output_dir / "compute_environment.json").write_text(json.dumps(env, indent=2) + "\n")
    xgboost_device = args.xgboost_device
    if xgboost_device == "auto":
        xgboost_device = "cuda" if env["gpus"] else "cpu"

    results = []
    for key in selected_model_keys(args.models):
        config = MODEL_CONFIGS[key]
        print(f"Training {key}...", flush=True)
        result = config.trainer(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            output_dir=output_dir,
            threshold_dir=threshold_dir,
            seed=args.seed,
            compress=args.compress,
            xgboost_device=xgboost_device,
        )
        result.update(
            {
                "artifact_name": config.artifact_name,
                "scaler_artifact_name": config.scaler_artifact_name,
                "config_notes": config.notes,
            }
        )
        results.append(result)
        pd.DataFrame(results).to_csv(output_dir / "training_summary_partial.csv", index=False)
        print(
            f"{key}: theta={result['pure_threshold']:.2f}, "
            f"strict={result['strict_accuracy']:.4f}, loose={result['loose_accuracy']:.4f}, "
            f"train={result['train_seconds']:.2f}s, infer={result['inference_seconds']:.2f}s",
            flush=True,
        )

    summary = pd.DataFrame(results).sort_values(["loose_accuracy", "strict_accuracy"], ascending=False)
    summary.to_csv(output_dir / "training_summary.csv", index=False)
    (output_dir / "training_summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2) + "\n")
    print(summary[["model", "pure_threshold", "strict_accuracy", "loose_accuracy", "train_seconds", "inference_seconds"]].to_string(index=False))


if __name__ == "__main__":
    main()
