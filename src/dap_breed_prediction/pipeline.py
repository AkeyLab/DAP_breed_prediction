import os
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
X_TRAIN_FILES = str(DATA_DIR / 'folder_of_54143_SNPs' / 'X_SNP_ch*_pruned_v3_std.parquet') # link will be provided
Y_TRAIN_FILE = str(DATA_DIR / 'y_combined_100.csv')
X_TRAIN_PCA_REPRODUCE = str(DATA_DIR / 'X_train_SNP_WG_prune_v3_1_std_pca_100.csv')
X_TEST_PCA_REPRODUCE = str(DATA_DIR / 'X_test_SNP_WG_prune_v3_1_std_pca_100.csv')
DEFAULT_SELECTED_BREEDS = [ 'Australian Shepherd',
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
# Backward-compatible alias.
DEFAULT_SELECED_BREEDS = DEFAULT_SELECTED_BREEDS
PCA_TRIGGER_PROPORTION = 0.35

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
        if os.path.exists(f'{result_folder_path}/Model/scaler.joblib'):
            logger.info(f'Loading the scaler at {result_folder_path}/Model/scaler.joblib')
            scaler = joblib.load(f'{result_folder_path}/Model/scaler.joblib')
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
        if os.path.exists(f'{result_folder_path}/Model/pca.joblib'):
            pca_model_path = f'{result_folder_path}/Model/pca.joblib'
            logger.info(f'Loading the pca model at {result_folder_path}/Model/pca.joblib')
            pca = joblib.load(f'{result_folder_path}/Model/pca.joblib')
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
                
                # trim the pca model
                pca.components_ = pca.components_[:pca_dim]
                pca.explained_variance_ = pca.explained_variance_[:pca_dim]
                pca.explained_variance_ratio_ = pca.explained_variance_ratio_[:pca_dim]
                pca.mean_ = pca.mean_
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
            pure_only = False, 
            force_recomputation = False
            ):
    # if the prediction model exists
    #matched_files = glob.glob(f"{result_folder_path}/Model/Prediction_model_theta_*.pkl")
    #if matched_files:    
    #    logger.info(f'The prediction model is found at {matched_files[0]}')
    #    return
    #if prediction_model_path is not None and os.path.exists(prediction_model_path):
    #    logger.info(f'The prediction model is found at {prediction_model_path}')
    #    return

    # if there is no prediction model

    # prepare Results folders
    os.makedirs(f'{result_folder_path}/Model', exist_ok=True)
    os.makedirs(f'{result_folder_path}/Table', exist_ok=True)
    os.makedirs(f'{result_folder_path}/Figure', exist_ok=True)

    # prepare the breed list    
    if breed_list_text_path is not None: # mode 2, 3
        with open(breed_list_text_path, "r") as f:
            selected_breeds = [line.strip() for line in f if line.strip()]
    else:
        if label_path is not None: # mode 3
            y_df = pd.read_csv(label_path)
            breed_set = set()
            for label in y_df['label']:
                parts = [p.strip() for p in label.split("/") if p.strip()]
                breed_set.update(parts) 
            selected_breeds = list(breed_set)
        else: # mode 1
            selected_breeds = DEFAULT_SELECTED_BREEDS # 14 pure breeds

    # construction DAP training data: mode 1, 2, 3, 5 

    # prepare Y_train
    Y_train = pd.read_csv(Y_TRAIN_FILE, index_col='dog_id')
    Y_train.index = Y_train.index.astype(str)
    Y_train.sort_index(inplace=True)

    if pure_only is False:
        if 'Unknown' not in selected_breeds:
            selected_breeds.append('Unknown')
        selected =  Y_train[selected_breeds].sum(axis=1) > 0
        not_others =  Y_train.drop(columns=selected_breeds).sum(axis=1) == 0
        Y_train =  Y_train[selected & not_others]
    else:
        selected_mask = (Y_train[selected_breeds] == 1.0).sum(axis=1) == 1
        non_selected_mask =  Y_train.drop(columns=selected_breeds).sum(axis=1) == 0
        Y_train =  Y_train[selected_mask & non_selected_mask]

    logger.info(f'The number of selected breeds is {len(selected_breeds)}')

    Y_train = Y_train[selected_breeds]
    logger.info(f'Y_train dim {Y_train.shape}')

    # prepare X_train
    dog_ids_train = Y_train.index
    dog_ids_train = dog_ids_train.astype(str)
    id_df = pl.DataFrame({"dog_id": list(dog_ids_train)})
    X_train_files = {helper.extract_chr_number(p): p for p in glob.glob(X_TRAIN_FILES)}
    
    header_df = pd.read_csv(SNP_csv_path, nrows=0)
    selected_snps = [c for c in header_df.columns if 'chr' in c]
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
    num_overlapped_snps = len(partial_dfs)
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
        theta_str = prediction_model_path.split("theta_")[1].replace(".pkl", "")
        theta = float(theta_str)
    else:
        pattern = f"{result_folder_path}/Model/Prediction_model_theta_*.pkl"
        matched_files = glob.glob(pattern)
        if matched_files:
            prediction_model_path = matched_files[0]
            logger.info(f'Loading the prediction model at {prediction_model_path}')
            theta_str = prediction_model_path.split("theta_")[1].replace(".pkl", "")
            theta = float(theta_str)
            regressor = joblib.load(prediction_model_path)
        else:
            logger.info(f'Training a prediction model') # only in this case
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.multioutput import MultiOutputRegressor
            regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators = 100, random_state = random_state, n_jobs = -1))
            regressor.fit(X_train, Y_train)
            Y_train_pred = regressor.predict(X_train)
            strict_accuracies, loose_accuracies = helper.pure_threshold_search(Y_train_pred, Y_train, mute = True)
            pure_thresholds = list(np.arange(0, 1.01, 0.01))
            theta = helper.best_pure_threshold_v1(pure_thresholds, strict_accuracies, loose_accuracies)
            theta = round(theta, 2)
            prediction_model_path = f'{result_folder_path}/Model/Prediction_model_theta_{theta}.pkl' 
            joblib.dump(regressor, prediction_model_path)
            logger.info(f'Best purity threshold (theta) is {theta}.')
            logger.info(f'Prediction model saved to {result_folder_path}/Model/Prediction_model_theta_{theta}.pkl\n')

    analyze.Generate_SNP_importance_score(result_folder_path, prediction_model_path, selected_snps, selected_breeds, pca_model_path)

def inference(  result_folder_path,  
                SNP_csv_path, # X_test
                label_path = None,  # Y_test
                breed_list_text_path = None, 
                scaler_path = None, # model
                pca_model_path = None, # model
                prediction_model_path = None, # model
                pca_components = None, # training param
                random_state = 42,
                pure_only = False, 
                force_recomputation = False
                ):

    # prepare X_test
    X_test = pd.read_csv(SNP_csv_path)
    X_test["dog_id"] = X_test["dog_id"].astype(str)

    # prepare the breed list    
    if breed_list_text_path is not None:
        with open(breed_list_text_path, "r") as f:
            selected_breeds = [line.strip() for line in f if line.strip()]
    elif label_path is not None:
        y_df = pd.read_csv(label_path)
        breed_set = set()
        for label in y_df['label']:
            parts = [p.strip() for p in label.split("/") if p.strip()]
            breed_set.update(parts) 
        selected_breeds = list(breed_set)    
    else:
        selected_breeds = DEFAULT_SELECTED_BREEDS
    
    if pure_only is False and 'Unknown' not in selected_breeds:
        selected_breeds.append('Unknown')

    logger.info(f'Inference(): There are {len(selected_breeds)} (m) pure breed classes.')
    
    # prepare Y_test (optional)
    Y_test = None
    if label_path is not None:
        breed_index = {b: i for i, b in enumerate(selected_breeds)}
        dog_order = X_test["dog_id"].tolist()
        label_df = pd.read_csv(label_path)
        label_df["dog_id"] = label_df["dog_id"].astype(str)
        id_to_label = dict(zip(label_df["dog_id"], label_df["label"]))
        n_samples = len(X_test)
        n_breeds = len(selected_breeds)
        Y_test = np.zeros((n_samples, n_breeds), dtype=float)
        for i, dog in enumerate(dog_order):
            label = id_to_label.get(dog, "")
            parts = helper.parse_label(label)
            if len(parts) == 1:
                Y_test[i, breed_index[parts[0]]] = 1.0
            elif len(parts) == 2:
                Y_test[i, breed_index[parts[0]]] = 0.5
                Y_test[i, breed_index[parts[1]]] = 0.5
        Y_test = pd.DataFrame(Y_test, index=dog_order, columns=selected_breeds)


        # generate the following results: Prediction map, per-class prediction 
    X_test.set_index("dog_id", inplace = True)
    x_test_ids = X_test.index.copy()
    logger.info(f'The dimension of X_test is {X_test.shape}')

    if scaler_path is None:
        scaler_path = f'{result_folder_path}/Model/scaler.joblib'
    if pca_model_path is None:
        pca_model_path = f'{result_folder_path}/Model/pca.joblib'    
    if os.path.exists(scaler_path) and os.path.exists(pca_model_path):
        logger.info(f'Loading the scaler at {scaler_path}')
        scaler = joblib.load(scaler_path)
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            index=X_test.index,
            columns=X_test.columns
        )
        logger.info(f'Loading the pca model at {pca_model_path}')
        pca = joblib.load(pca_model_path)
        X_test = pd.DataFrame(
            pca.transform(X_test_scaled),
            index=x_test_ids
        )
    
    if prediction_model_path is not None:
        logger.info(f'Loading the prediction model at {prediction_model_path}')
        regressor = joblib.load(prediction_model_path)
        theta_str = prediction_model_path.split("theta_")[1].replace(".pkl", "")
        theta = float(theta_str)
    else:
        pattern = f"{result_folder_path}/Model/Prediction_model_theta_*.pkl"
        matched_files = glob.glob(pattern)
        if matched_files:
            model_path = matched_files[0]
            logger.info(f'Loading the prediction model at {model_path}')
            theta_str = model_path.split("theta_")[1].replace(".pkl", "")
            theta = float(theta_str)
            regressor = joblib.load(model_path)
        else:
            # no prediction model is found. There should be an error
            logger.info('No prediction model is found.')
            return

    # run inference / prediction
    Y_pred_raw = regressor.predict(X_test) # Y_pred_raw is numpy array
    
    # save raw predictions
    if os.path.exists(f'{result_folder_path}/Table/Raw_prediction.csv') and not force_recomputation:
        logger.info(f'Raw prediction already at {result_folder_path}/Table/Raw_prediction.csv')
        Y_pred_raw_df = pd.read_csv(f'{result_folder_path}/Table/Raw_prediction.csv', index_col = 'dog_id')
        Y_pred_raw_df.index = Y_pred_raw_df.index.astype(str)
    else:
        Y_pred_raw_df = pd.DataFrame(Y_pred_raw, index = x_test_ids, columns=selected_breeds)
        Y_pred_raw_df.to_csv(f'{result_folder_path}/Table/Raw_prediction.csv')
        logger.info(f'Raw prediction saved to {result_folder_path}/Table/Raw_prediction.csv')

    # save transformed predictions
    if os.path.exists(f'{result_folder_path}/Table/Transformed_prediction.csv') and not force_recomputation:
        logger.info(f'Transformed prediction already at {result_folder_path}/Table/Transformed_prediction.csv')
        Y_pred_transformed_df = pd.read_csv(f'{result_folder_path}/Table/Transformed_prediction.csv', index_col = 'dog_id')
        Y_pred_transformed_df.index = Y_pred_transformed_df.index.astype(str)
    else:
        Y_pred_transformed = helper.transform_prediction(Y_pred_raw, theta) # Y_pred_transformed is numpy array
        Y_pred_transformed_df = pd.DataFrame(Y_pred_transformed, index = x_test_ids, columns=selected_breeds)
        Y_pred_transformed_df.to_csv(f'{result_folder_path}/Table/Transformed_prediction.csv')
        logger.info(f'Transformed prediction saved to {result_folder_path}/Table/Transformed_prediction.csv')

    # save prediction labels
    if os.path.exists(f'{result_folder_path}/Table/Predictions.csv') and not force_recomputation:
        logger.info(f'Predictions (text label) already at {result_folder_path}/Table/Predictions.csv')
    else:
        Y_pred_labels = Y_pred_transformed_df.apply(helper.reconstruct_label, axis=1)
        Y_pred_label_df = pd.DataFrame(Y_pred_labels, index = x_test_ids, columns=['Prediction'])
        Y_pred_label_df.to_csv(f'{result_folder_path}/Table/Predictions.csv')
        logger.info(f'Predictions (text label) saved to {result_folder_path}/Table/Predictions.csv')

    if Y_test is not None: # need input labels
        strict_acc,loose_acc,metadata = helper.prediction_analysis(Y_pred_raw, Y_test, pure_threshold=theta, mute = False)
        helper.plot_confusion_matrix_highlighted(Y_test, Y_pred_raw, theta, metadata, save_path = result_folder_path, save_plot = True)


def full_training_pipeline( result_folder_path,  
                            SNP_csv_path = None, # X_test
                            label_path = None,  # Y_test
                            breed_list_text_path = None, 
                            pca_components = None, # training param
                            random_state = 42,
                            test_size = 0.3,
                            pure_only = False):
    # Mode 4: X and Y must be provided
    # Mode 5: breed list must be provided
    # Mode 6: breed list must be passed as "reproduce"
        # prepare Results folders
    os.makedirs(f'{result_folder_path}/Model', exist_ok=True)
    os.makedirs(f'{result_folder_path}/Table', exist_ok=True)
    os.makedirs(f'{result_folder_path}/Figure', exist_ok=True)
    # prepare X and Y
    if breed_list_text_path == 'reproduce': # mode 6
        y_combined  = pd.read_csv(Y_TRAIN_FILE, index_col='dog_id')
        y_combined.index = y_combined.index.astype(str)
        y_combined.sort_index(inplace=True)
        selected_breeds = list(y_combined.columns)
        X_train_df = pd.read_csv(X_TRAIN_PCA_REPRODUCE, index_col = 'dog_id')
        X_test_df = pd.read_csv(X_TEST_PCA_REPRODUCE, index_col = 'dog_id')
        X_train_df.index = X_train_df.index.astype(str)
        X_test_df.index = X_test_df.index.astype(str)
        X_train_df.sort_index(inplace=True)
        X_test_df.sort_index(inplace=True)
        Y_train = y_combined.loc[y_combined.index.intersection(X_train_df.index)]
        Y_test  = y_combined.loc[y_combined.index.intersection(X_test_df.index)]
        X_train = X_train_df.loc[Y_train.index]
        X_test  = X_test_df.loc[Y_test.index]
        selected_snps = X_train.columns.tolist()
    else:
        if SNP_csv_path is None:
            # mode 5
            # get breeds from DAP X and DAP Y
            with open(breed_list_text_path, "r") as f:
                selected_breeds = [line.strip() for line in f if line.strip()]

            Y = pd.read_csv(Y_TRAIN_FILE, index_col='dog_id')
            Y.index = Y.index.astype(str)
            Y.sort_index(inplace=True)
            if pure_only is False:
                if 'Unknown' not in selected_breeds:
                    selected_breeds.append('Unknown')
                selected =  Y[selected_breeds].sum(axis=1) > 0
                not_others =  Y.drop(columns=selected_breeds).sum(axis=1) == 0
                Y =  Y[selected & not_others]
            else:
                selected_mask = (Y[selected_breeds] == 1.0).sum(axis=1) == 1
                non_selected_mask =  Y.drop(columns=selected_breeds).sum(axis=1) == 0
                Y =  Y[selected_mask & non_selected_mask]
            Y = Y[selected_breeds]
            logger.info(f'Y dim {Y.shape}')
            
            dog_ids_train = Y.index
            dog_ids_train = dog_ids_train.astype(str)
            id_df = pl.DataFrame({"dog_id": list(dog_ids_train)})
            X_files = sorted(glob.glob(X_TRAIN_FILES), key=helper.extract_chr_number)
            dfs_X = [pl.read_parquet(f) for f in X_files]
            dog_id_X = dfs_X[0]["dog_id"]
            dfs_X = [df.drop("dog_id") if i > 0 else df for i, df in enumerate(dfs_X)]
            X_full = pl.concat(dfs_X, how="horizontal")
            X = X_full.to_pandas()
            X.set_index("dog_id", inplace=True)
            selected_snps = X.columns.tolist()
            X = X.loc[Y.index]
            assert X.index.equals(Y.index)

        else:
            # mode 4
            # construct X and Y from SNP_csv_path and label_path
            X = pd.read_csv(SNP_csv_path, index_col = 'dog_id')
            X.index = X.index.astype(str)
            selected_snps = X.columns.tolist()

            label_df = pd.read_csv(label_path)
            breed_set = set()
            for label in label_df['label']:
                parts = [p.strip() for p in label.split("/") if p.strip()]
                breed_set.update(parts) 
            selected_breeds = list(breed_set)

            if breed_list_text_path is not None:
                with open(breed_list_text_path, "r") as f:
                    selected_breeds = [line.strip() for line in f if line.strip()]
            logger.info(f'There are {len(selected_breeds)} (m) pure breed classes.')
            breed_index = {b: i for i, b in enumerate(selected_breeds)}
            dog_order = X.index.tolist()
            label_df = pd.read_csv(label_path)
            label_df["dog_id"] = label_df["dog_id"].astype(str)
            id_to_label = dict(zip(label_df["dog_id"], label_df["label"]))
            n_samples = len(dog_order)
            n_breeds = len(selected_breeds)
            Y = np.zeros((n_samples, n_breeds), dtype=float)
            for i, dog in enumerate(dog_order):
                label = id_to_label.get(dog, "")
                parts = helper.parse_label(label)
                if len(parts) == 1:
                    Y[i, breed_index[parts[0]]] = 1.0
                elif len(parts) == 2:
                    Y[i, breed_index[parts[0]]] = 0.5
                    Y[i, breed_index[parts[1]]] = 0.5
            Y = pd.DataFrame(Y, index=dog_order, columns=selected_breeds)
            assert X.index.equals(Y.index)
        # safe split X, Y --> X_train, X_test, Y_train, Y_test
        X_train, X_test, Y_train, Y_test, _ = helper.safe_train_test_split(X, Y, test_size=test_size, random_state=random_state)

        assert X_train.index.equals(Y_train.index)
        assert X_test.index.equals(Y_test.index)

        num_training_samples, num_overlapped_snps = X_train.shape
        # decide if PCA is needed
        pca_model_path = None
        if pca_components is not None or (num_overlapped_snps / num_training_samples) > PCA_TRIGGER_PROPORTION:
            if pca_components == 'None':
                pca_components = 0.95
            X_train, pca_model_path = apply_pca(X_train = X_train, result_folder_path = result_folder_path, pca_components = pca_components, num_training_samples = num_training_samples)
            # scaler saved to {result_folder_path}/Model/scaler.joblib
            # pca model saved to {result_folder_path}/Model/pca.joblib
            # transform X_test
            scaler_model = joblib.load(f'{result_folder_path}/Model/scaler.joblib')
            pca_model = joblib.load(f'{result_folder_path}/Model/pca.joblib')
            X_test_scaled = pd.DataFrame(
                scaler_model.transform(X_test),
                index=X_test.index,
                columns=X_test.columns
            )
            X_test = pd.DataFrame(
                pca_model.transform(X_test_scaled),
                index=X_test_scaled.index
            )

    x_test_ids = X_test.index.copy()

    pattern = f"{result_folder_path}/Model/Prediction_model_theta_*.pkl"
    matched_files = glob.glob(pattern)
    if matched_files:
        prediction_model_path = matched_files[0]
        logger.info(f'Loading the prediction model at {prediction_model_path}')
        theta_str = prediction_model_path.split("theta_")[1].replace(".pkl", "")
        theta = float(theta_str)
        regressor = joblib.load(prediction_model_path)
    else:
        logger.info(f'Training a prediction model') # only in this case
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators = 100, random_state = random_state, n_jobs = -1))
        regressor.fit(X_train, Y_train)
        Y_train_pred = regressor.predict(X_train)
        strict_accuracies, loose_accuracies = helper.pure_threshold_search(Y_train_pred, Y_train, mute = True)
        pure_thresholds = list(np.arange(0, 1.01, 0.01))
        theta = helper.best_pure_threshold_v1(pure_thresholds, strict_accuracies, loose_accuracies)
        theta = round(theta, 2)
        joblib.dump(regressor, f'{result_folder_path}/Model/Prediction_model_theta_{theta}.pkl')
        prediction_model_path = f'{result_folder_path}/Model/Prediction_model_theta_{theta}.pkl'
        logger.info(f'Best purity threshold (theta) is {theta}.')
        logger.info(f'Prediction model saved to {result_folder_path}/Model/Prediction_model_theta_{theta}.pkl\n')
    
    # run inference / prediction
    Y_pred_raw = regressor.predict(X_test) # Y_pred_raw is numpy array
    
    # save raw predictions
    if os.path.exists(f'{result_folder_path}/Table/Raw_prediction.csv'):
        logger.info(f'Raw prediction already at {result_folder_path}/Table/Raw_prediction.csv')
        Y_pred_raw_df = pd.read_csv(f'{result_folder_path}/Table/Raw_prediction.csv', index_col = 'dog_id')
        Y_pred_raw_df.index = Y_pred_raw_df.index.astype(str)
    else:
        Y_pred_raw_df = pd.DataFrame(Y_pred_raw, index = x_test_ids, columns=selected_breeds)
        Y_pred_raw_df.to_csv(f'{result_folder_path}/Table/Raw_prediction.csv')
        logger.info(f'Raw prediction saved to {result_folder_path}/Table/Raw_prediction.csv')

    # save transformed predictions
    if os.path.exists(f'{result_folder_path}/Table/Transformed_prediction.csv'):
        logger.info(f'Transformed prediction already at {result_folder_path}/Table/Transformed_prediction.csv')
        Y_pred_transformed_df = pd.read_csv(f'{result_folder_path}/Table/Transformed_prediction.csv', index_col = 'dog_id')
        Y_pred_transformed_df.index = Y_pred_transformed_df.index.astype(str)
    else:
        Y_pred_transformed = helper.transform_prediction(Y_pred_raw, theta) # Y_pred_transformed is numpy array
        Y_pred_transformed_df = pd.DataFrame(Y_pred_transformed, index = x_test_ids, columns=selected_breeds)
        Y_pred_transformed_df.to_csv(f'{result_folder_path}/Table/Transformed_prediction.csv')
        logger.info(f'Transformed prediction saved to {result_folder_path}/Table/Transformed_prediction.csv')

    # save prediction labels
    if os.path.exists(f'{result_folder_path}/Table/Predictions.csv'):
        logger.info(f'Predictions (text label) already at {result_folder_path}/Table/Predictions.csv')
    else:
        Y_pred_labels = Y_pred_transformed_df.apply(helper.reconstruct_label, axis=1)
        Y_pred_label_df = pd.DataFrame(Y_pred_labels, index = x_test_ids, columns=['Prediction'])
        Y_pred_label_df.to_csv(f'{result_folder_path}/Table/Predictions.csv')
        logger.info(f'Predictions (text label) saved to {result_folder_path}/Table/Predictions.csv')

    strict_acc,loose_acc,metadata = helper.prediction_analysis(Y_pred_raw, Y_test, pure_threshold=theta, mute = False)
    helper.plot_confusion_matrix_highlighted(Y_test, Y_pred_raw, theta, metadata, save_path = result_folder_path, save_plot = False)

    if breed_list_text_path == 'reproduce':
        return
    analyze.Generate_SNP_importance_score(result_folder_path, prediction_model_path, selected_snps, selected_breeds, pca_model_path)
