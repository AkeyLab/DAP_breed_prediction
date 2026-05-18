import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm


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

def Generate_SNP_importance_score(result_folder_path, prediction_model_path, snp_names, selected_breeds = None, pca_model_path = None):
    if selected_breeds is None:
        selected_breeds = DEFAULT_SELECTED_BREEDS

    prediction_model = joblib.load(prediction_model_path)
    estimators = prediction_model.estimators_  # list of RandomForestRegressor(), one per output
    feature_importances_matrix = np.vstack([est.feature_importances_ for est in estimators]) # shape: (n_outputs, n_PCs) w PCA or (n_outputs, n_SNPs) w/o PCA
    
    if pca_model_path is not None:
        pca_model = joblib.load(pca_model_path)
        components = pca_model.components_  # shape: (n_components, n_SNPs)
        explained_var_ratio = pca_model.explained_variance_ratio_  # shape: (n_components,)
        weighted_pc_importances = feature_importances_matrix * explained_var_ratio  
        snp_importances_matrix = np.dot(weighted_pc_importances, np.abs(components))  # shape: (n_outputs, n_SNPs)
    else:
        # rename for consistency
        snp_importances_matrix = feature_importances_matrix

    assert len(selected_breeds) == snp_importances_matrix.shape[0]

    per_class_importance_df = pd.DataFrame(
    snp_importances_matrix,
    index=selected_breeds,
    columns=snp_names
    ).reset_index().melt(id_vars='index', var_name='snp', value_name='importance')

    # Rename for clarity
    per_class_importance_df.rename(columns={'index': 'breed'}, inplace=True)

    per_breed_dir = f"{result_folder_path}/Table/Snp_importance_per_breed"
    os.makedirs(per_breed_dir, exist_ok=True)

    for breed in selected_breeds:
        df_breed = (
            per_class_importance_df
            .query("breed == @breed")
            .sort_values("importance", ascending= False)
            #.head(top_k)
            .copy()
        )
        df_breed['rank'] = np.arange(1, len(df_breed)+1)
        df_breed['breed'] = breed

        filename = f"{per_breed_dir}/Snp_importance_{breed.replace(' ', '_')}.csv"
        if not os.path.exists(filename):
            df_breed.to_csv(filename, index=False)
    plot_importance_by_rank(result_folder_path)


def plot_importance_by_rank(result_folder_path):
    folder = Path(f'{result_folder_path}/Table/Snp_importance_per_breed')
    plt.figure(figsize=(10, 10))

    file_data = []

    # Collect valid data
    for csv_file in folder.glob("*.csv"):
        df = pd.read_csv(csv_file)
        if 'rank' in df.columns and 'importance' in df.columns:
            df_sorted = df.sort_values(by='rank')
            file_data.append((csv_file.stem, df_sorted))

    # Sort file names alphabetically
    file_data.sort(key=lambda x: x[0])
    n_files = len(file_data)
    # Generate combined colors from two colormaps
    half = (n_files + 1) // 2
    colors1 = cm.get_cmap('tab10', half)(np.linspace(0, 1, half))
    colors2 = cm.get_cmap('Dark2', n_files - half)(np.linspace(0, 1, n_files - half))
    combined_colors = np.vstack([colors1, colors2])

    for idx, (file_name, df_sorted) in enumerate(file_data):
        label_name = file_name.replace('Snp_importance_','').replace('.csv','')
        plt.plot(df_sorted['rank'], df_sorted['importance'], label=label_name, color=combined_colors[idx])

    plt.xlabel("Rank")
    plt.legend(title="Breed", loc='best')
    #plt.xlim(0, 55000)  # Set x-axis range
    #plt.ylim(0, 0.00025)  # Set x-axis range
    
    plt.ylabel("Snp Importance")
    plt.savefig(f'{result_folder_path}/Figure/SNP_importance.svg', format="svg", transparent=True)
