import os
import json
import subprocess
import pandas as pd
import numpy as np
import polars as pl
import joblib
import glob
from functools import reduce
from pathlib import Path
from . import helpers as helper
from . import analysis as analyze
import logging
logger = logging.getLogger(__name__)

# GLOBAL VARIABLES
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
X_TRAIN_FILES = str(DATA_DIR / 'folder_of_54143_SNPs' / 'X_SNP_ch*_pruned_v3_std.parquet')
Y_TRAIN_FILE = str(DATA_DIR / 'y_combined_100.csv')
X_TRAIN_PCA_REPRODUCE = str(DATA_DIR / 'X_train_SNP_WG_prune_v3_1_std_pca_100.csv')
X_TEST_PCA_REPRODUCE = str(DATA_DIR / 'X_test_SNP_WG_prune_v3_1_std_pca_100.csv')
PRETRAINED_PCA_FILE = str(MODEL_DIR / 'pca_model_WG_100.joblib')
PRETRAINED_RF_FILE = str(MODEL_DIR / 'regressor_model0_4-PCA100.pkl')
PRETRAINED_MODEL_REGISTRY_FILE = MODEL_DIR / 'model_registry.json'
PAPER_14_BREEDS = [ 'Australian Shepherd',
                    'Beagle',
                    'Bernese Mountain Dog',
                    'Border Collie',
                    'Boston Terrier',
                    'Cavalier King Charles Spaniel',
                    'Dachshund',
                    'French Bulldog',
                    'German Shepherd Dog',
                    'Golden Retriever',
                    'Great Dane',
                    'Labrador Retriever',
                    'Pembroke Welsh Corgi',
                    'Poodle',
                    ]
# Backward-compatible aliases for code that imported the former constant.
DEFAULT_SELECTED_BREEDS = PAPER_14_BREEDS
DEFAULT_SELECED_BREEDS = PAPER_14_BREEDS
PCA_TRIGGER_PROPORTION = 0.35
DEFAULT_INFERENCE_PURE_THRESHOLD = 0.7
FULL_DAP_SNP_COUNT = 54143
MODEL_METADATA_FILENAME = 'model_metadata.json'
MODE4_TRAINING_MODELS = {
    'random_forest',
    'xgboost',
    'ridge',
    'knn',
    'extratrees',
    'mlp',
    'transformer',
}


def _normalize_model_name(name):
    if name is None:
        return None
    return str(name).strip().lower().replace('-', '_').replace(' ', '_')


def _normalize_training_model_name(name):
    name = _normalize_model_name(name) or 'random_forest'
    aliases = {
        'rf': 'random_forest',
        'randomforest': 'random_forest',
        'random_forest': 'random_forest',
        'xgb': 'xgboost',
        'xgboost': 'xgboost',
        'extra_trees': 'extratrees',
        'extra_trees_regressor': 'extratrees',
        'extratrees': 'extratrees',
        'ridge': 'ridge',
        'knn': 'knn',
        'mlp': 'mlp',
        'transformer': 'transformer',
    }
    normalized = aliases.get(name, name)
    if normalized not in MODE4_TRAINING_MODELS:
        raise ValueError(
            f'Unknown training_model_name {name!r}. '
            f'Available models: {sorted(MODE4_TRAINING_MODELS)}'
        )
    return normalized


def _load_pretrained_model_registry():
    if not PRETRAINED_MODEL_REGISTRY_FILE.is_file():
        return {
            'default_model': 'random_forest',
            'models': {
                'random_forest': {
                    'display_name': 'Random Forest',
                    'model_type': 'sklearn',
                    'model_path': Path(PRETRAINED_RF_FILE).name,
                    'scaler_path': None,
                    'pure_threshold': DEFAULT_INFERENCE_PURE_THRESHOLD,
                }
            },
        }
    registry = json.loads(PRETRAINED_MODEL_REGISTRY_FILE.read_text())
    if 'models' not in registry or not isinstance(registry['models'], dict):
        raise ValueError(f'Invalid pretrained model registry: {PRETRAINED_MODEL_REGISTRY_FILE}')
    return registry


def list_pretrained_models():
    """Return bundled pretrained model metadata keyed by model name."""
    return dict(_load_pretrained_model_registry()['models'])


def _resolve_model_path(recorded_path):
    if recorded_path is None:
        return None
    path = Path(recorded_path)
    if not path.is_absolute():
        path = MODEL_DIR / path
    return str(path)


def _resolve_pretrained_model_config(
    pretrained_model_name=None,
    prediction_model_path=None,
    prediction_model_type=None,
    scaler_path=None,
    pure_threshold=None,
):
    registry = _load_pretrained_model_registry()
    model_name = _normalize_model_name(pretrained_model_name)
    if prediction_model_path is None:
        model_name = model_name or registry.get('default_model', 'random_forest')
        models = registry['models']
        aliases = {
            _normalize_model_name(key): key for key in models
        }
        if model_name not in aliases:
            raise ValueError(
                f'Unknown pretrained_model_name {pretrained_model_name!r}. '
                f'Available models: {sorted(models)}'
            )
        registry_key = aliases[model_name]
        config = dict(models[registry_key])
        config['registry_key'] = registry_key
        config['prediction_model_path'] = _resolve_model_path(config.get('model_path'))
        config['scaler_path'] = _resolve_model_path(config.get('scaler_path'))
        if pure_threshold is None:
            pure_threshold = config.get('pure_threshold')
    else:
        config = {
            'registry_key': model_name or 'custom',
            'display_name': pretrained_model_name or 'Custom model',
            'model_type': prediction_model_type or 'sklearn',
            'prediction_model_path': str(prediction_model_path),
            'scaler_path': str(scaler_path) if scaler_path is not None else None,
        }
    config['model_type'] = prediction_model_type or config.get('model_type', 'sklearn')
    if scaler_path is not None:
        config['scaler_path'] = str(scaler_path)
    config['pure_threshold'] = normalize_pure_threshold(pure_threshold)
    if config['pure_threshold'] is None:
        raise ValueError(
            'pure_threshold must be supplied for custom Mode 1 models, or the '
            'selected bundled model must define pure_threshold in model_registry.json.'
        )
    return config


def _optional_dependency_error(model_type, exc):
    missing = getattr(exc, 'name', None) or str(exc)
    if missing == 'xgboost':
        package = 'xgboost'
    elif missing == 'torch' or model_type.startswith('torch_'):
        package = 'torch'
    else:
        package = missing
    raise ImportError(
        f'The selected pretrained model requires optional dependency {package!r}. '
        'Install it with `python -m pip install -e ".[alternative-models]"` '
        'or choose the random_forest model.'
    ) from exc


def _make_torch_mlp(torch, n_features, n_outputs):
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


def _make_torch_transformer(torch, n_features, n_outputs):
    class PCTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            d_model = 64
            self.value_embed = torch.nn.Linear(1, d_model)
            self.pos_embed = torch.nn.Parameter(torch.zeros(1, n_features, d_model))
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=128,
                dropout=0.15,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
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


def _predict_torch_model(model_path, model_type, X_model, n_outputs):
    try:
        import torch
    except ModuleNotFoundError as exc:
        _optional_dependency_error(model_type, exc)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    n_features = X_model.shape[1]
    if model_type == 'torch_mlp':
        model = _make_torch_mlp(torch, n_features, n_outputs)
    elif model_type == 'torch_transformer':
        model = _make_torch_transformer(torch, n_features, n_outputs)
    else:
        raise ValueError(f'Unsupported torch pretrained model type: {model_type}')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    arr = X_model.to_numpy(dtype='float32')
    preds = []
    with torch.no_grad():
        for start in range(0, len(arr), 1024):
            xb = torch.from_numpy(arr[start:start + 1024]).to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds.append(probs.detach().cpu().numpy())
    return np.vstack(preds)


def _predict_pretrained_model(model_path, model_type, X_model, n_outputs):
    if model_type == 'sklearn':
        try:
            model = joblib.load(model_path)
        except ModuleNotFoundError as exc:
            _optional_dependency_error(model_type, exc)
        predict_input = X_model
        if getattr(model, 'feature_names_in_', None) is None:
            predict_input = X_model.to_numpy()
        return np.asarray(model.predict(predict_input))
    if model_type.startswith('torch_'):
        return _predict_torch_model(model_path, model_type, X_model, n_outputs)
    raise ValueError(f'Unsupported pretrained model type: {model_type}')


def get_all_breed_classes():
    """Return all 100 output classes in the label-table column order."""
    return list(pd.read_csv(Y_TRAIN_FILE, index_col='dog_id', nrows=0).columns)


def configure_unknown_class(selected_breeds, include_unknown):
    """Return a copied class list with the requested Unknown-class behavior."""
    selected_breeds = [breed for breed in selected_breeds if breed != 'Unknown']
    if include_unknown:
        selected_breeds.append('Unknown')
    return selected_breeds


def normalize_pure_threshold(pure_threshold):
    """Validate and normalize an optional pure-versus-mixed threshold."""
    if pure_threshold is None:
        return None
    if isinstance(pure_threshold, bool):
        raise ValueError("pure_threshold must be a number between 0 and 1.")
    try:
        pure_threshold = float(pure_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("pure_threshold must be a number between 0 and 1.") from exc
    if not np.isfinite(pure_threshold) or not 0 <= pure_threshold <= 1:
        raise ValueError("pure_threshold must be a number between 0 and 1.")
    return round(pure_threshold, 2)


def _prepare_result_directories(result_folder_path):
    for directory in ('Model', 'Table', 'Figure'):
        os.makedirs(f'{result_folder_path}/{directory}', exist_ok=True)


def _save_prediction_outputs(result_folder_path, sample_ids, class_names, raw, theta):
    raw_df = pd.DataFrame(raw, index=sample_ids, columns=class_names)
    raw_df.index.name = 'dog_id'
    raw_df.to_csv(f'{result_folder_path}/Table/Raw_prediction.csv')
    logger.info(f'Raw prediction saved to {result_folder_path}/Table/Raw_prediction.csv')

    transformed = helper.transform_prediction(raw_df.to_numpy(), theta)
    transformed_df = pd.DataFrame(
        transformed, index=sample_ids, columns=class_names
    )
    transformed_df.index.name = 'dog_id'
    transformed_df.to_csv(f'{result_folder_path}/Table/Transformed_prediction.csv')
    logger.info(f'Transformed prediction saved to {result_folder_path}/Table/Transformed_prediction.csv')

    labels = transformed_df.apply(helper.reconstruct_label, axis=1)
    labels_df = labels.to_frame(name='Prediction')
    labels_df.index.name = 'dog_id'
    labels_df.to_csv(f'{result_folder_path}/Table/Predictions.csv')
    logger.info(f'Predictions (text label) saved to {result_folder_path}/Table/Predictions.csv')
    return raw_df, transformed_df, labels_df


def _write_model_metadata(
    result_folder_path,
    prediction_model_path,
    selected_snps,
    selected_breeds,
    pure_threshold,
    pca_model_path=None,
    scaler_path=None,
    model_input_scaler_path=None,
    model_type='sklearn',
    model_name='random_forest',
    training_source='combined_dap_reference',
):
    metadata = {
        'schema_version': 2,
        'training_source': training_source,
        'model_name': model_name,
        'model_type': model_type,
        'prediction_model_path': Path(prediction_model_path).name,
        'pure_threshold': pure_threshold,
        'class_names': list(selected_breeds),
        'snp_names': list(selected_snps),
        'pca_model_path': Path(pca_model_path).name if pca_model_path else None,
        'scaler_path': Path(scaler_path).name if scaler_path else None,
        'model_input_scaler_path': (
            Path(model_input_scaler_path).name if model_input_scaler_path else None
        ),
    }
    metadata_path = Path(result_folder_path) / 'Model' / MODEL_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    logger.info(f'Model metadata saved to {metadata_path}')
    return metadata_path


def _load_model_metadata(model_metadata_path, prediction_model_path):
    if model_metadata_path is None:
        model_metadata_path = Path(prediction_model_path).parent / MODEL_METADATA_FILENAME
    metadata_path = Path(model_metadata_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f'Model metadata not found at {metadata_path}. Models trained by '
            'Mode 2 or Mode 4 save this sidecar automatically.'
        )
    metadata = json.loads(metadata_path.read_text())
    required = {'class_names', 'snp_names'}
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f'Model metadata is missing required fields: {missing}')
    return metadata, metadata_path


def _resolve_model_sidecar(metadata_path, configured_path, recorded_name):
    if configured_path is not None:
        return str(configured_path)
    if not recorded_name:
        return None
    path = Path(recorded_name)
    if not path.is_absolute():
        path = metadata_path.parent / path
    return str(path)


def _validate_exact_snp_list(actual, expected):
    actual_set = set(actual)
    expected_set = set(expected)
    missing = [name for name in expected if name not in actual_set]
    extra = [name for name in actual if name not in expected_set]
    if missing or extra:
        raise ValueError(
            'Input SNP list does not match the tested model. '
            f'Expected {len(expected):,}, received {len(actual):,}; '
            f'missing {len(missing):,}, extra {len(extra):,}. '
            f'First missing: {missing[:3]}; first extra: {extra[:3]}.'
        )


def pretrained_inference(
    result_folder_path,
    SNP_csv_path,
    pca_model_path=None,
    prediction_model_path=None,
    pure_threshold=None,
    pretrained_model_name=None,
    prediction_model_type=None,
    scaler_path=None,
):
    """Run the bundled paper PCA and a selected 100-output model on full SNP input."""
    _prepare_result_directories(result_folder_path)
    model_config = _resolve_pretrained_model_config(
        pretrained_model_name=pretrained_model_name,
        prediction_model_path=prediction_model_path,
        prediction_model_type=prediction_model_type,
        scaler_path=scaler_path,
        pure_threshold=pure_threshold,
    )
    theta = model_config['pure_threshold']
    pca_model_path = pca_model_path or PRETRAINED_PCA_FILE
    prediction_model_path = model_config['prediction_model_path']
    scaler_path = model_config.get('scaler_path')
    model_type = model_config.get('model_type', 'sklearn')
    display_name = model_config.get('display_name', model_config.get('registry_key', 'model'))

    X = pd.read_csv(SNP_csv_path, index_col='dog_id')
    X.index = X.index.astype(str)
    if len(X) != 1:
        raise ValueError(
            f'Mode 1 requires exactly one sample row; received {len(X):,}.'
        )
    pca = joblib.load(pca_model_path)
    expected_snps = [str(name) for name in pca.feature_names_in_]
    actual_snps = X.columns.tolist()
    _validate_exact_snp_list(actual_snps, expected_snps)
    if actual_snps != expected_snps:
        logger.info('Input SNP columns were reordered to match the pretrained PCA.')
        X = X.loc[:, expected_snps]

    logger.info(f'Input X contains {len(X):,} sample(s) and {len(actual_snps):,} SNPs.')
    logger.info(f'Full pretrained input requirement satisfied: {len(expected_snps):,} SNPs.')
    logger.info(f'Loading the pretrained PCA at {pca_model_path}')
    pcs = pca.transform(X)

    pc_names = [f'PC{i}' for i in range(1, pcs.shape[1] + 1)]
    X_pca = pd.DataFrame(pcs, index=X.index, columns=pc_names)

    X_model = X_pca
    if scaler_path is not None:
        logger.info(f'Loading the pretrained PC scaler at {scaler_path}')
        scaler = joblib.load(scaler_path)
        scaler_features = list(getattr(scaler, 'feature_names_in_', X_pca.columns))
        if list(X_pca.columns) != scaler_features:
            X_model = X_pca.loc[:, scaler_features]
        X_model = pd.DataFrame(
            scaler.transform(X_model),
            index=X_pca.index,
            columns=scaler_features,
        )

    class_names = get_all_breed_classes()
    logger.info(
        f'Loading pretrained {display_name} ({model_type}) at {prediction_model_path}'
    )
    raw = _predict_pretrained_model(
        prediction_model_path, model_type, X_model, len(class_names)
    )
    if raw.shape[1] != len(class_names):
        raise ValueError(
            f'Pretrained model produced {raw.shape[1]} outputs; expected {len(class_names)}.'
        )
    logger.info(f'Using pretrained {display_name} purity threshold (theta) {theta}.')
    _save_prediction_outputs(result_folder_path, X.index, class_names, raw, theta)


def apply_pca(  X_train,
                result_folder_path,
                pca_components = 0.95,
                scaler_path = None,
                pca_model_path = None,
                num_training_samples = None):
    
    if pca_components is None:
        pca_components = 0.95

    if scaler_path is not None:
        logger.info(f'Loading the scaler at {scaler_path}')
        scaler = joblib.load(scaler_path)
        X_train_scaled = pd.DataFrame(
            scaler.transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        )
    else:
        logger.info(f'Training a scaler')
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        )
        joblib.dump(scaler, f'{result_folder_path}/Model/scaler.joblib')
        logger.info(f'scaler saved to {result_folder_path}/Model/scaler.joblib\n')
    
    if pca_model_path is not None:
        logger.info(f'Loading the pca model at {pca_model_path}')
        pca = joblib.load(pca_model_path)
        X_train = pca.transform(X_train_scaled)
    else:
        logger.info(f'Training a pca model')
        from sklearn.decomposition import PCA
        pca = PCA(n_components = pca_components, svd_solver='auto')
        X_train = pca.fit_transform(X_train_scaled)
        pca_dim = pca.n_components_
        if pca_dim > num_training_samples * PCA_TRIGGER_PROPORTION:
            pca_dim = int(num_training_samples * PCA_TRIGGER_PROPORTION)
            logger.info(f'PCA dim ({pca_dim}) is still too large. Only keeping top {pca_dim} PCs. The pca model will also be trimmed.')
            X_train = X_train[:, :pca_dim]

            # Keep the fitted transformer consistent with the retained scores.
            pca.components_ = pca.components_[:pca_dim]
            pca.explained_variance_ = pca.explained_variance_[:pca_dim]
            pca.explained_variance_ratio_ = pca.explained_variance_ratio_[:pca_dim]
            pca.n_components_ = pca_dim

        joblib.dump(pca, f'{result_folder_path}/Model/pca.joblib')
        logger.info(f'pca model saved to {result_folder_path}/Model/pca.joblib\n')
        pca_model_path = f'{result_folder_path}/Model/pca.joblib'
    return X_train, pca_model_path

def train(  result_folder_path,  
            SNP_csv_path, # X_test
            label_path = None,  # Y_test
            breed_list_text_path = None, 
            scaler_path = None, # model
            pca_model_path = None, # model
            prediction_model_path = None, # model
            pca_components = None, # training param
            random_state = 42,
            pure_threshold = None,
            pure_only = False, 
            include_unknown = True,
            force_recomputation = False
            ):
    configured_theta = normalize_pure_threshold(pure_threshold)
    _prepare_result_directories(result_folder_path)

    # An optional class list narrows the 100 DAP outputs used for retraining.
    if breed_list_text_path is not None:
        with open(breed_list_text_path, "r") as f:
            selected_breeds = [line.strip() for line in f if line.strip()]
    else:
        if label_path is not None:
            y_df = pd.read_csv(label_path)
            breed_set = set()
            for label in y_df['label']:
                parts = [p.strip() for p in label.split("/") if p.strip()]
                breed_set.update(parts) 
            selected_breeds = list(breed_set)
        else:
            selected_breeds = None

    # Mode 2 deliberately combines all labeled DAP reference partitions.
    Y_train = pd.read_csv(Y_TRAIN_FILE, index_col='dog_id')
    Y_train.index = Y_train.index.astype(str)
    Y_train.sort_index(inplace=True)
    logger.info(
        f'Complete combined DAP reference: {len(Y_train):,} samples, '
        f'{len(Y_train.columns):,} classes, {FULL_DAP_SNP_COUNT:,} SNPs.'
    )

    if selected_breeds is None:
        selected_breeds = list(Y_train.columns)

    if pure_only is False:
        selected_breeds = configure_unknown_class(selected_breeds, include_unknown)
        selected =  Y_train[selected_breeds].sum(axis=1) > 0
        not_others =  Y_train.drop(columns=selected_breeds).sum(axis=1) == 0
        Y_train =  Y_train[selected & not_others]
    else:
        selected_mask = (Y_train[selected_breeds] == 1.0).sum(axis=1) == 1
        non_selected_mask =  Y_train.drop(columns=selected_breeds).sum(axis=1) == 0
        Y_train =  Y_train[selected_mask & non_selected_mask]

    logger.info(f'The number of selected breeds is {len(selected_breeds)}')
    logger.info(f'Selected breed classes: {selected_breeds}')

    Y_train = Y_train[selected_breeds]
    logger.info(f'Y_train dim {Y_train.shape}')

    # prepare X_train
    dog_ids_train = Y_train.index
    dog_ids_train = dog_ids_train.astype(str)
    id_df = pl.DataFrame({"dog_id": list(dog_ids_train)})
    X_train_files = {helper.extract_chr_number(p): p for p in glob.glob(X_TRAIN_FILES)}
    
    header_df = pd.read_csv(SNP_csv_path, nrows=0)
    selected_snps = [c for c in header_df.columns if 'chr' in c]
    input_sample_count = len(pd.read_csv(SNP_csv_path, usecols=['dog_id']))
    logger.info(
        f'Input X contains {input_sample_count:,} sample(s) and '
        f'{len(selected_snps):,} requested SNPs.'
    )
    chr_to_snps = helper.group_snps_by_chr(selected_snps)
    partial_dfs = []
    for chr_num, cols in chr_to_snps.items():
        path = X_train_files.get(chr_num)
        if path is None:
            raise FileNotFoundError(f"No parquet file found for chr{chr_num}")

        # Read schema-only to see which requested columns actually exist
        cols_in_file = pl.read_parquet(path, n_rows=0).columns
        wanted = ["dog_id"] + [c for c in cols if c in cols_in_file]
        if len(wanted) <= 1:
            # No requested SNPs present in this chromosome file
            continue

        # Read only the needed columns, then keep only requested dog_ids
        df_chr = pl.read_parquet(path, columns=wanted)
        df_chr = id_df.join(df_chr, on="dog_id", how="inner")  # filters rows to your sample set

        partial_dfs.append(df_chr)
    if not partial_dfs:
        raise ValueError("None of the selected SNPs were found in the parquet files.")

    # Join all chromosome chunks on dog_id
    X_pl = reduce(lambda left, right: left.join(right, on="dog_id", how="inner"), partial_dfs)

    # Reorder rows and columns to match your requested order
    X_pl = id_df.join(X_pl, on="dog_id", how="left")  # preserve dog_ids_train order
    X_df = X_pl.to_pandas().set_index("dog_id")

    # Keep exactly the requested SNP columns (in your given order)
    existing = [c for c in selected_snps if c in X_df.columns]
    X_train = X_df.loc[dog_ids_train, existing]

    num_training_samples, num_overlapped_snps = X_train.shape

    logger.info(f'The dimension of X_train is {X_train.shape}.')
    logger.info(f'Thus, there are {num_training_samples} training samples and {num_overlapped_snps} provided SNPs are overlapped with 54,143 SNPs used in the paper.')
    logger.info(
        f'Overlapping SNP set X\' contains {num_overlapped_snps:,} SNPs. '
        'All eligible samples from the combined DAP train and test partitions are used.'
    )

    # if there are too many features (SNPs), apply PCA
    if (num_overlapped_snps / num_training_samples) > PCA_TRIGGER_PROPORTION:
        logger.info(f'There are too many features (SNPs) compared to the number of training samples. PCA will be applied.')         
        if pca_components is None:
            pca_components = 0.95
        X_train, pca_model_path = apply_pca(X_train,
                result_folder_path,
                pca_components = pca_components,
                scaler_path = None,
                pca_model_path = None,
                num_training_samples = num_training_samples)
    
    if prediction_model_path is not None:
        logger.info(f'Loading the prediction model at {prediction_model_path}')
        regressor = joblib.load(prediction_model_path)
        if configured_theta is None:
            theta_str = prediction_model_path.split("theta_")[1].replace(".pkl", "")
            theta = float(theta_str)
        else:
            theta = configured_theta
    else:
        logger.info(f'Training a prediction model')
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators = 100, random_state = random_state, n_jobs = -1))
        regressor.fit(X_train, Y_train)
        if configured_theta is None:
            Y_train_pred = regressor.predict(X_train)
            strict_accuracies, loose_accuracies = helper.pure_threshold_search(Y_train_pred, Y_train, mute = True)
            pure_thresholds = list(np.arange(0, 1.01, 0.01))
            theta = helper.best_pure_threshold_v1(pure_thresholds, strict_accuracies, loose_accuracies)
            theta = round(theta, 2)
            logger.info(f'Best purity threshold (theta) is {theta}.')
        else:
            theta = configured_theta
            logger.info(f'Using configured purity threshold (theta) {theta}.')
        prediction_model_path = f'{result_folder_path}/Model/Prediction_model_theta_{theta}.pkl'
        joblib.dump(regressor, prediction_model_path)
        logger.info(f'Prediction model saved to {result_folder_path}/Model/Prediction_model_theta_{theta}.pkl\n')

    selected_snps = existing
    Path(f'{result_folder_path}/Table/Overlapping_SNPs.txt').write_text(
        ''.join(f'{name}\n' for name in selected_snps)
    )
    Path(f'{result_folder_path}/Table/Selected_classes.txt').write_text(
        ''.join(f'{name}\n' for name in selected_breeds)
    )
    scaler_path = None
    if pca_model_path is not None:
        candidate = Path(result_folder_path) / 'Model' / 'scaler.joblib'
        if candidate.is_file():
            scaler_path = str(candidate)
    metadata_path = _write_model_metadata(
        result_folder_path,
        prediction_model_path,
        selected_snps,
        selected_breeds,
        theta,
        pca_model_path=pca_model_path,
        scaler_path=scaler_path,
    )
    analyze.Generate_SNP_importance_score(result_folder_path, prediction_model_path, selected_snps, selected_breeds, pca_model_path)
    return {
        'prediction_model_path': Path(prediction_model_path),
        'model_metadata_path': metadata_path,
        'pure_threshold': theta,
    }

def inference(
    result_folder_path,
    SNP_csv_path,
    prediction_model_path=None,
    pure_threshold=None,
    label_path=None,
    model_metadata_path=None,
    scaler_path=None,
    pca_model_path=None,
    model_input_scaler_path=None,
    prediction_model_type=None,
    require_exact_features=True,
):
    """Run a Mode 2/4 model with metadata-backed feature and class validation."""
    _prepare_result_directories(result_folder_path)

    if prediction_model_path is None:
        candidates = sorted(glob.glob(
            f'{result_folder_path}/Model/Prediction_model_theta_*.pkl'
        ))
        candidates.extend(sorted(glob.glob(
            f'{result_folder_path}/Model/Prediction_model_*_theta_*'
        )))
        if not candidates:
            raise FileNotFoundError('No prediction model was provided or found.')
        prediction_model_path = candidates[0]

    model_metadata, metadata_path = _load_model_metadata(
        model_metadata_path, prediction_model_path
    )
    class_names = list(model_metadata['class_names'])
    expected_snps = list(model_metadata['snp_names'])
    theta = normalize_pure_threshold(pure_threshold)
    if theta is None:
        theta = normalize_pure_threshold(model_metadata.get('pure_threshold'))
    if theta is None:
        raise ValueError('pure_threshold must be provided for model inference.')

    X = pd.read_csv(SNP_csv_path, index_col='dog_id')
    X.index = X.index.astype(str)
    actual_snps = X.columns.tolist()
    if require_exact_features:
        _validate_exact_snp_list(actual_snps, expected_snps)
    else:
        missing = [name for name in expected_snps if name not in X.columns]
        if missing:
            raise ValueError(
                f'Input is missing {len(missing):,} SNPs required by the retrained model: '
                f'{missing[:3]}.'
            )
        ignored = len(actual_snps) - len(expected_snps)
        if ignored:
            logger.info(f'Ignoring {ignored:,} input SNPs outside overlapping set X\'.')
    if actual_snps != expected_snps:
        X = X.loc[:, expected_snps]

    logger.info(
        f'Validated X against model metadata: {len(X):,} sample(s), '
        f'{len(expected_snps):,} exact model SNPs, {len(class_names):,} output classes.'
    )
    scaler_path = _resolve_model_sidecar(
        metadata_path, scaler_path, model_metadata.get('scaler_path')
    )
    pca_model_path = _resolve_model_sidecar(
        metadata_path, pca_model_path, model_metadata.get('pca_model_path')
    )
    model_input_scaler_path = _resolve_model_sidecar(
        metadata_path,
        model_input_scaler_path,
        model_metadata.get('model_input_scaler_path')
    )
    model_type = prediction_model_type or model_metadata.get('model_type', 'sklearn')
    model_name = model_metadata.get('model_name', 'prediction model')

    X_model = X
    if scaler_path is not None:
        logger.info(f'Loading the model scaler at {scaler_path}')
        scaler = joblib.load(scaler_path)
        X_model = pd.DataFrame(
            scaler.transform(X_model), index=X.index, columns=expected_snps
        )
    if pca_model_path is not None:
        logger.info(f'Loading the model PCA at {pca_model_path}')
        pca = joblib.load(pca_model_path)
        transformed = pca.transform(X_model)
        columns = [f'PC{i}' for i in range(1, transformed.shape[1] + 1)]
        X_model = pd.DataFrame(transformed, index=X.index, columns=columns)
    if model_input_scaler_path is not None:
        logger.info(f'Loading the model-input scaler at {model_input_scaler_path}')
        model_input_scaler = joblib.load(model_input_scaler_path)
        scaler_features = list(getattr(model_input_scaler, 'feature_names_in_', X_model.columns))
        if list(X_model.columns) != scaler_features:
            X_model = X_model.loc[:, scaler_features]
        X_model = pd.DataFrame(
            model_input_scaler.transform(X_model),
            index=X_model.index,
            columns=scaler_features,
        )

    logger.info(f'Loading the {model_name} model at {prediction_model_path}')
    raw = _predict_pretrained_model(
        prediction_model_path, model_type, X_model, len(class_names)
    )
    if raw.shape[1] != len(class_names):
        raise ValueError(
            f'Model produced {raw.shape[1]} outputs but metadata defines '
            f'{len(class_names)} classes.'
        )
    logger.info(f'Using supplied purity threshold (theta) {theta}.')
    _save_prediction_outputs(result_folder_path, X.index, class_names, raw, theta)

    performance = None
    if label_path is not None:
        labels = pd.read_csv(label_path)
        required_columns = {'dog_id', 'label'}
        if not required_columns.issubset(labels.columns):
            raise ValueError('Label CSV must contain dog_id and label columns.')
        labels['dog_id'] = labels['dog_id'].astype(str)
        if labels['dog_id'].duplicated().any():
            raise ValueError('Label CSV contains duplicate dog_id values.')
        label_by_id = labels.set_index('dog_id')['label']
        missing_labels = [dog_id for dog_id in X.index if dog_id not in label_by_id.index]
        if missing_labels:
            raise ValueError(f'Labels are missing for input samples: {missing_labels[:5]}')

        breed_index = {breed: index for index, breed in enumerate(class_names)}
        Y_test = np.zeros((len(X), len(class_names)), dtype=float)
        for row_index, dog_id in enumerate(X.index):
            label = label_by_id.loc[dog_id]
            parts = helper.parse_label(label)
            if not parts and str(label).strip() == 'Unknown' and 'Unknown' in breed_index:
                parts = ['Unknown']
            if len(parts) not in (1, 2):
                raise ValueError(f'Unsupported label for dog {dog_id}: {label!r}')
            unknown_classes = [part for part in parts if part not in breed_index]
            if unknown_classes:
                raise ValueError(
                    f'Label for dog {dog_id} contains classes absent from the model: '
                    f'{unknown_classes}'
                )
            value = 1.0 if len(parts) == 1 else 0.5
            for part in parts:
                Y_test[row_index, breed_index[part]] = value
        Y_test = pd.DataFrame(Y_test, index=X.index, columns=class_names)

        strict_accuracy, loose_accuracy, per_sample = helper.prediction_analysis(
            raw, Y_test, pure_threshold=theta, mute=False
        )
        performance = pd.DataFrame([{
            'n_samples': len(X),
            'pure_threshold': theta,
            'strict_accuracy': strict_accuracy,
            'loose_accuracy': loose_accuracy,
        }])
        performance.to_csv(
            f'{result_folder_path}/Table/Performance_metrics.csv', index=False
        )
        per_sample.index.name = 'dog_id'
        per_sample.to_csv(f'{result_folder_path}/Table/Per_sample_performance.csv')
        helper.plot_confusion_matrix_highlighted(
            Y_test,
            raw,
            theta,
            per_sample,
            save_path=result_folder_path,
            save_plot=True,
        )
        logger.info(
            f'Performance metrics saved to '
            f'{result_folder_path}/Table/Performance_metrics.csv'
        )

    return {
        'prediction_model_path': Path(prediction_model_path),
        'model_metadata_path': metadata_path,
        'pure_threshold': theta,
        'performance': performance,
    }


def _xgboost_device(device):
    device = _normalize_model_name(device) or 'auto'
    if device not in {'auto', 'cpu', 'cuda'}:
        raise ValueError("xgboost_device must be one of: auto, cpu, cuda.")
    if device != 'auto':
        return device
    try:
        subprocess.run(
            ['nvidia-smi'],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return 'cuda'
    except (OSError, subprocess.CalledProcessError):
        return 'cpu'


def _pc_columns(width):
    return [f'PC{i}' for i in range(1, width + 1)]


def _scale_model_input_if_needed(
    training_model_name,
    X_train,
    X_test,
    result_folder_path,
):
    if training_model_name not in {'ridge', 'knn', 'mlp', 'transformer'}:
        return X_train, X_test, None
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns,
    )
    path = f'{result_folder_path}/Model/{training_model_name}_model_input_scaler.joblib'
    joblib.dump(scaler, path)
    logger.info(f'Model-input scaler saved to {path}')
    return X_train_scaled, X_test_scaled, path


def _make_mode4_sklearn_model(training_model_name, random_state, xgboost_device, n_train):
    if training_model_name == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        return MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1,
            )
        )
    if training_model_name == 'xgboost':
        try:
            import xgboost as xgb
        except ModuleNotFoundError as exc:
            _optional_dependency_error('xgboost', exc)
        return xgb.XGBRegressor(
            objective='reg:squarederror',
            multi_strategy='one_output_per_tree',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            tree_method='hist',
            device=_xgboost_device(xgboost_device),
            random_state=random_state,
            n_jobs=-1,
        )
    if training_model_name == 'ridge':
        from sklearn.linear_model import Ridge
        return Ridge(alpha=10.0)
    if training_model_name == 'knn':
        from sklearn.neighbors import KNeighborsRegressor
        n_neighbors = max(1, min(15, n_train))
        return KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance', n_jobs=-1)
    if training_model_name == 'extratrees':
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(
            n_estimators=300,
            max_features=0.75,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f'{training_model_name} is not an sklearn Mode 4 model.')


def _fit_torch_mode4_model(
    training_model_name,
    X_train,
    X_test,
    Y_train,
    random_state,
):
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        _optional_dependency_error(f'torch_{training_model_name}', exc)

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.benchmark = False

    X_train_arr = X_train.to_numpy(dtype='float32')
    X_test_arr = X_test.to_numpy(dtype='float32')
    Y_train_arr = Y_train.to_numpy(dtype='float32')
    if len(X_train_arr) >= 8:
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            np.arange(len(X_train_arr)),
            test_size=0.15,
            random_state=random_state,
        )
    else:
        train_idx = np.arange(len(X_train_arr))
        val_idx = np.arange(len(X_train_arr))

    train_ds = TensorDataset(
        torch.from_numpy(X_train_arr[train_idx]),
        torch.from_numpy(Y_train_arr[train_idx]),
    )
    val_x = torch.from_numpy(X_train_arr[val_idx]).to(device)
    val_y = torch.from_numpy(Y_train_arr[val_idx]).to(device)
    loader = DataLoader(train_ds, batch_size=min(256, max(1, len(train_ds))), shuffle=True)

    if training_model_name == 'mlp':
        model = _make_torch_mlp(torch, X_train.shape[1], Y_train.shape[1]).to(device)
        lr = 1e-3
        epochs = 220
        model_type = 'torch_mlp'
    elif training_model_name == 'transformer':
        model = _make_torch_transformer(torch, X_train.shape[1], Y_train.shape[1]).to(device)
        lr = 8e-4
        epochs = 220
        model_type = 'torch_transformer'
    else:
        raise ValueError(f'{training_model_name} is not a torch Mode 4 model.')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def soft_target_ce(logits, target):
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        return -(target * log_probs).sum(dim=1).mean()

    patience = 30
    best_state = None
    best_val = float('inf')
    best_epoch = -1
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = soft_target_ce(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(soft_target_ce(model(val_x), val_y).detach().cpu())
        history.append({
            'epoch': epoch,
            'train_loss': float(np.mean(train_losses)),
            'val_loss': val_loss,
        })
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    def predict(arr):
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(arr), 1024):
                xb = torch.from_numpy(arr[start:start + 1024]).to(device)
                probs = torch.softmax(model(xb), dim=1)
                preds.append(probs.detach().cpu().numpy())
        return np.vstack(preds)

    checkpoint = {
        'model_name': training_model_name,
        'model_state_dict': model.state_dict(),
        'best_epoch': best_epoch,
        'best_val_loss': best_val,
    }
    return {
        'model_type': model_type,
        'checkpoint': checkpoint,
        'train_raw': predict(X_train_arr),
        'test_raw': predict(X_test_arr),
        'history': pd.DataFrame(history),
    }


def _fit_mode4_prediction_model(
    training_model_name,
    X_train,
    X_test,
    Y_train,
    result_folder_path,
    random_state,
    xgboost_device='auto',
):
    training_model_name = _normalize_training_model_name(training_model_name)
    X_train_model, X_test_model, model_input_scaler_path = _scale_model_input_if_needed(
        training_model_name, X_train, X_test, result_folder_path
    )
    if training_model_name in {'mlp', 'transformer'}:
        torch_result = _fit_torch_mode4_model(
            training_model_name,
            X_train_model,
            X_test_model,
            Y_train,
            random_state,
        )
        history_path = f'{result_folder_path}/Table/{training_model_name}_training_history.csv'
        torch_result['history'].to_csv(history_path, index=False)
        logger.info(f'{training_model_name} training history saved to {history_path}')
        return {
            'model': torch_result['checkpoint'],
            'model_type': torch_result['model_type'],
            'train_raw': torch_result['train_raw'],
            'test_raw': torch_result['test_raw'],
            'model_input_scaler_path': model_input_scaler_path,
            'extension': 'pt',
        }

    model = _make_mode4_sklearn_model(
        training_model_name,
        random_state,
        xgboost_device,
        len(X_train_model),
    )
    predict_input = X_train_model
    if training_model_name in {'ridge', 'knn'}:
        predict_input = X_train_model.to_numpy()
    model.fit(predict_input, Y_train)

    train_predict_input = predict_input
    test_predict_input = X_test_model
    if training_model_name in {'ridge', 'knn'}:
        test_predict_input = X_test_model.to_numpy()
    return {
        'model': model,
        'model_type': 'sklearn',
        'train_raw': np.asarray(model.predict(train_predict_input)),
        'test_raw': np.asarray(model.predict(test_predict_input)),
        'model_input_scaler_path': model_input_scaler_path,
        'extension': 'pkl',
    }


def _save_mode4_prediction_model(result_folder_path, training_model_name, theta, fit_result):
    if training_model_name == 'random_forest':
        path = f'{result_folder_path}/Model/Prediction_model_theta_{theta}.pkl'
    else:
        path = (
            f'{result_folder_path}/Model/'
            f'Prediction_model_{training_model_name}_theta_{theta}.{fit_result["extension"]}'
        )
    if fit_result['model_type'].startswith('torch_'):
        try:
            import torch
        except ModuleNotFoundError as exc:
            _optional_dependency_error(fit_result['model_type'], exc)
        torch.save(fit_result['model'], path)
    else:
        joblib.dump(fit_result['model'], path)
    logger.info(f'Prediction model saved to {path}')
    return path


def full_training_pipeline(
    result_folder_path,
    SNP_csv_path=None,
    label_path=None,
    breed_list_text_path=None,
    pca_components=None,
    random_state=42,
    test_size=0.3,
    training_model_name='random_forest',
    xgboost_device='auto',
):
    """Train/evaluate user data (Mode 4) or reproduce the paper split (Mode 5)."""
    _prepare_result_directories(result_folder_path)
    reproduce = breed_list_text_path == 'reproduce'
    training_model_name = 'random_forest' if reproduce else _normalize_training_model_name(training_model_name)
    pca_model_path = None
    scaler_path = None
    model_input_scaler_path = None

    if reproduce:
        y_combined = pd.read_csv(Y_TRAIN_FILE, index_col='dog_id')
        y_combined.index = y_combined.index.astype(str)
        y_combined.sort_index(inplace=True)
        selected_breeds = list(y_combined.columns)
        X_train_df = pd.read_csv(X_TRAIN_PCA_REPRODUCE, index_col='dog_id')
        X_test_df = pd.read_csv(X_TEST_PCA_REPRODUCE, index_col='dog_id')
        X_train_df.index = X_train_df.index.astype(str)
        X_test_df.index = X_test_df.index.astype(str)
        X_train_df.sort_index(inplace=True)
        X_test_df.sort_index(inplace=True)
        Y_train = y_combined.loc[y_combined.index.intersection(X_train_df.index)]
        Y_test = y_combined.loc[y_combined.index.intersection(X_test_df.index)]
        X_train = X_train_df.loc[Y_train.index]
        X_test = X_test_df.loc[Y_test.index]
        selected_snps = X_train.columns.tolist()
    else:
        if SNP_csv_path is None or label_path is None:
            raise ValueError('Mode 4 requires both SNP_csv_path and label_path.')
        X = pd.read_csv(SNP_csv_path, index_col='dog_id')
        X.index = X.index.astype(str)
        selected_snps = X.columns.tolist()

        labels = pd.read_csv(label_path)
        if not {'dog_id', 'label'}.issubset(labels.columns):
            raise ValueError('Label CSV must contain dog_id and label columns.')
        labels['dog_id'] = labels['dog_id'].astype(str)
        label_by_id = labels.set_index('dog_id')['label']
        missing_labels = [dog_id for dog_id in X.index if dog_id not in label_by_id.index]
        if missing_labels:
            raise ValueError(f'Labels are missing for input samples: {missing_labels[:5]}')

        if breed_list_text_path is None:
            selected_breeds = sorted({
                breed
                for label in labels['label']
                for breed in helper.parse_label(label)
            })
        else:
            with open(breed_list_text_path) as handle:
                selected_breeds = [line.strip() for line in handle if line.strip()]
        logger.info(f'User dataset contains {len(X):,} samples and {len(X.columns):,} SNPs.')
        logger.info(f'User-defined output classes ({len(selected_breeds)}): {selected_breeds}')

        breed_index = {breed: index for index, breed in enumerate(selected_breeds)}
        Y_values = np.zeros((len(X), len(selected_breeds)), dtype=float)
        for row_index, dog_id in enumerate(X.index):
            label = label_by_id.loc[dog_id]
            parts = helper.parse_label(label)
            if len(parts) not in (1, 2):
                raise ValueError(f'Unsupported label for dog {dog_id}: {label!r}')
            unknown_classes = [part for part in parts if part not in breed_index]
            if unknown_classes:
                raise ValueError(
                    f'Label for dog {dog_id} contains classes absent from the class list: '
                    f'{unknown_classes}'
                )
            value = 1.0 if len(parts) == 1 else 0.5
            for part in parts:
                Y_values[row_index, breed_index[part]] = value
        Y = pd.DataFrame(Y_values, index=X.index, columns=selected_breeds)

        X_train, X_test, Y_train, Y_test, _ = helper.safe_train_test_split(
            X, Y, test_size=test_size, random_state=random_state
        )
        num_training_samples, num_features = X_train.shape
        if pca_components is not None or (
            num_features / num_training_samples
        ) > PCA_TRIGGER_PROPORTION:
            if pca_components in (None, 'None'):
                pca_components = 0.95
            X_train, pca_model_path = apply_pca(
                X_train=X_train,
                result_folder_path=result_folder_path,
                pca_components=pca_components,
                num_training_samples=num_training_samples,
            )
            X_train = pd.DataFrame(
                X_train,
                index=Y_train.index,
                columns=_pc_columns(X_train.shape[1]),
            )
            scaler_path = f'{result_folder_path}/Model/scaler.joblib'
            scaler = joblib.load(scaler_path)
            pca = joblib.load(pca_model_path)
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test), index=X_test.index, columns=X_test.columns
            )
            X_test_values = pca.transform(X_test_scaled)
            X_test = pd.DataFrame(
                X_test_values,
                index=X_test_scaled.index,
                columns=_pc_columns(X_test_values.shape[1]),
            )

    x_test_ids = X_test.index.copy()
    logger.info(f'Training a {training_model_name} prediction model')
    fit_result = _fit_mode4_prediction_model(
        training_model_name,
        X_train,
        X_test,
        Y_train,
        result_folder_path,
        random_state,
        xgboost_device=xgboost_device,
    )
    model_input_scaler_path = fit_result['model_input_scaler_path']
    train_raw = fit_result['train_raw']
    strict_accuracies, loose_accuracies = helper.pure_threshold_search(
        train_raw, Y_train, mute=True
    )
    thresholds = list(np.arange(0, 1.01, 0.01))
    theta = round(helper.best_pure_threshold_v1(
        thresholds, strict_accuracies, loose_accuracies
    ), 2)
    prediction_model_path = _save_mode4_prediction_model(
        result_folder_path,
        training_model_name,
        theta,
        fit_result,
    )
    logger.info(f'Best purity threshold (theta) is {theta}.')

    raw = fit_result['test_raw']
    _save_prediction_outputs(
        result_folder_path, x_test_ids, selected_breeds, raw, theta
    )
    strict_accuracy, loose_accuracy, per_sample = helper.prediction_analysis(
        raw, Y_test, pure_threshold=theta, mute=False
    )
    pd.DataFrame([{
        'n_samples': len(X_test),
        'training_model_name': training_model_name,
        'model_type': fit_result['model_type'],
        'pure_threshold': theta,
        'strict_accuracy': strict_accuracy,
        'loose_accuracy': loose_accuracy,
    }]).to_csv(f'{result_folder_path}/Table/Performance_metrics.csv', index=False)
    per_sample.index.name = 'dog_id'
    per_sample.to_csv(f'{result_folder_path}/Table/Per_sample_performance.csv')
    helper.plot_confusion_matrix_highlighted(
        Y_test,
        raw,
        theta,
        per_sample,
        save_path=result_folder_path,
        save_plot=False,
    )

    if reproduce:
        return {
            'prediction_model_path': Path(prediction_model_path),
            'pure_threshold': theta,
        }

    metadata_path = _write_model_metadata(
        result_folder_path,
        prediction_model_path,
        selected_snps,
        selected_breeds,
        theta,
        pca_model_path=pca_model_path,
        scaler_path=scaler_path,
        model_input_scaler_path=model_input_scaler_path,
        model_type=fit_result['model_type'],
        model_name=training_model_name,
        training_source='user_supplied',
    )
    if training_model_name == 'random_forest':
        analyze.Generate_SNP_importance_score(
            result_folder_path,
            prediction_model_path,
            selected_snps,
            selected_breeds,
            pca_model_path,
        )
    else:
        logger.info(
            'Skipping SNP importance generation because it is currently implemented '
            'for random forest Mode 4 models only.'
        )
    return {
        'prediction_model_path': Path(prediction_model_path),
        'model_metadata_path': metadata_path,
        'pure_threshold': theta,
    }
