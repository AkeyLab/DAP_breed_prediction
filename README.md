# DAP Breed Prediction

Code and data package for dog breed prediction/admixture estimation used in the DAP paper workflows: "***An interpretable machine learning framework for dog breed inference and ancestry decomposition***".

## Repository Layout

```text
DAP_breed_prediction/
├── main.py                              # CLI entrypoint
├── pyproject.toml                       # Package metadata
├── requirements.txt                     # Dependency list
├── configs/                             # YAML config templates for modes 1-6
├── data/                                # Toy data + reproduction assets
│   ├── Toy_X_snps.csv
│   ├── Toy_Y_labels.csv
│   ├── Toy_a_short_breed_list.txt
│   ├── X_train_SNP_WG_prune_v3_1_std_pca_100.csv
│   ├── X_test_SNP_WG_prune_v3_1_std_pca_100.csv
│   └── y_combined_100.csv
└── src/dap_breed_prediction/
    ├── cli.py                           # Mode parsing + orchestration
    ├── pipeline.py                      # Training/inference pipeline
    ├── helpers.py                       # Metrics, plotting, data split helpers
    └── analysis.py                      # SNP importance analysis
```

## Installation

```bash
pip install -r requirements.txt
```

Optional editable install (enables the `dap-breed-predict` command):

```bash
pip install -e .
```

## Usage

Run from repository root:

```bash
python main.py <mode flags> -config_path <path_to_config.yml>
```

Or, if installed with `pip install -e .`:

```bash
dap-breed-predict <mode flags> -config_path <path_to_config.yml>
```

## Modes

### Mode 1
Train on DAP data, infer on provided SNPs, fixed default breed panel.

- Flags: `-inference`
- Required config keys:
  - `result_folder_path`
  - `SNP_csv_path`

### Mode 2
Train on DAP data, infer on provided SNPs, breed set from provided breed list.

- Flags: `-inference`
- Required config keys:
  - `result_folder_path`
  - `SNP_csv_path`
  - `breed_list_text_path`

### Mode 3
Train on DAP data, infer on provided SNPs, and evaluate with labels.

- Flags: `-inference`
- Required config keys:
  - `result_folder_path`
  - `SNP_csv_path`
  - `label_path`
- Optional:
  - `breed_list_text_path` (if supplied, it is prioritized for class set definition)

### Mode 4
Train/test split on provided dataset (no DAP SNP parquet dependency).

- Flags: `-train -inference`
- Required config keys:
  - `result_folder_path`
  - `SNP_csv_path`
  - `label_path`
- Optional:
  - `breed_list_text_path`
  - `pca_components` (default `0.95`)
  - `random_state` (default `42`)
  - `test_size` (default `0.3`)

### Mode 5
Train on DAP data using a user-provided breed list.

- Flags: `-train`
- Required config keys:
  - `result_folder_path`
  - `breed_list_text_path`
- Optional:
  - `pca_components` (default `0.95`)
  - `random_state` (default `42`)
  - `test_size` (default `0.3`)

### Mode 6
Reproduce the 100-class paper model using bundled PCA-space train/test matrices.

- Flags: `-reproduce`
- Required config keys:
  - `result_folder_path`

## Config Templates

Ready-to-edit templates are in `configs/`:

- `config_mode_1_template.yml`
- `config_mode_2_template.yml`
- `config_mode_3_template.yml`
- `config_mode_4_template.yml`
- `config_mode_5_template.yml`
- `config_mode_6_template.yml`

## Input File Expectations

- `SNP_csv_path` (CSV):
  - Must include a `dog_id` column.
  - SNP columns should be named like `chr<chromosome>:...`.
- `label_path` (CSV):
  - Must include `dog_id` and `label`.
  - Label format:
    - Pure: `BreedName`
    - Mixed: `BreedA / BreedB`
- `breed_list_text_path` (TXT):
  - One breed name per line.

## Outputs

Each run writes to `<result_folder_path>/`:

- `process.log`
- `Model/`:
  - `Prediction_model_theta_<value>.pkl`
  - optional `scaler.joblib`, `pca.joblib`
- `Table/`:
  - `Raw_prediction.csv`
  - `Transformed_prediction.csv`
  - `Predictions.csv`
  - optional per-section/per-class performance tables
  - optional SNP-importance tables
- `Figure/`:
  - optional prediction map and SNP-importance SVGs

## Important Data Note for Modes 1/2/3/5

Modes `1/2/3/5` require chromosome-wise DAP SNP parquet files at:

```text
data/folder_of_54143_SNPs/X_SNP_ch*_pruned_v3_std.parquet
```

These parquet files are not bundled in this repository and must be provided separately.

## Quick Smoke Test (Mode 4 With Toy Data)

```bash
python main.py -train -inference -config_path configs/config_mode_4_template.yml
```

