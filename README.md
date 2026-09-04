# DAP Breed Prediction

Code and data package for dog breed prediction/admixture estimation used in the DAP paper workflows: "***An interpretable machine learning framework for dog breed inference and ancestry decomposition***".

## Repository Layout

```text
DAP_breed_prediction/
├── main.py                              # CLI entrypoint
├── pyproject.toml                       # Package metadata
├── requirements.txt                     # Dependency list
├── configs/                             # YAML config templates for modes 1-6
├── Figure_data/                          # Inputs for the executed figure notebook
├── data/                                # Toy data + reproduction assets
│   ├── Toy_X_snps.csv
│   ├── Toy_Y_labels.csv
│   ├── Toy_a_short_breed_list.txt
│   ├── X_train_SNP_WG_prune_v3_1_std_pca_100.csv
│   ├── X_test_SNP_WG_prune_v3_1_std_pca_100.csv
│   └── y_combined_100.csv
├── notebooks/
│   └── reproduce_selected_paper_figures.ipynb
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

## Hardware And Runtime

- No non-standard hardware is required; CPU-only execution is sufficient.
- Typical install time: Typically <5 minutes on a normal desktop with internet access.
- Expected demo runtime: Typically <1 minute for the toy dataset after dependencies are installed.

## Reproduce The Paper Results

Mode 6 reproduces the paper's 100-class breed-prediction experiment. It trains a new random-forest model with the paper settings (`100` PCA components and random seed `42`), selects the pure-versus-mixed prediction threshold on the training set, and evaluates the model on the fixed test set.

The required PCA-space data and labels are included in this repository, so this workflow does **not** require the external chromosome-wise parquet files used by Modes 1, 2, 3, and 5.

### 1. Clone The Repository

```bash
git clone https://github.com/AkeyLab/DAP_breed_prediction.git
cd DAP_breed_prediction
```

### 2. Create An Environment And Install Dependencies

Python `3.9` or newer is required. A virtual environment is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Run The Reproduction

From the repository root, run:

```bash
python main.py -reproduce -config_path configs/config_mode_6_template.yml
```

No configuration changes are needed for a first run. The supplied config writes results to `results/mode_6/`. Training is CPU-only and takes approximately 4-5 minutes on the system used for the reference run; runtime will vary with CPU resources.

Mode 6 uses these bundled files:

- `data/X_train_SNP_WG_prune_v3_1_std_pca_100.csv`: training samples represented by 100 principal components
- `data/X_test_SNP_WG_prune_v3_1_std_pca_100.csv`: fixed test samples represented by the same components
- `data/y_combined_100.csv`: 100-class breed/admixture labels

The separately bundled pretrained model in `model/` is not used by this command; Mode 6 trains a new model to reproduce the training and evaluation procedure.

### 4. Verify The Results

A successful reference run reports:

```text
Best purity threshold (theta) is 0.7.
Strict Accuracy: 58.67%
Loose Accuracy: 91.71%
Number of Wrong Samples - Strict: 818, Loose: 164
```

Strict accuracy requires the complete pure/mixed breed assignment to match. Loose accuracy counts a prediction as correct when at least one predicted breed overlaps the true assignment. Small numerical differences may occur across operating systems or dependency versions.

The run creates:

```text
results/mode_6/
├── process.log
├── Model/
│   └── Prediction_model_theta_0.7.pkl
├── Table/
│   ├── Raw_prediction.csv
│   ├── Transformed_prediction.csv
│   ├── Predictions.csv
│   ├── Per_section_performance.csv
│   └── Per_class_prediction_details.csv
└── Figure/
```

Check `results/mode_6/process.log` for the parameters, threshold, and summary metrics from the run. To perform another independent clean run, copy `configs/config_mode_6_template.yml`, change `result_folder_path` to a new directory, and run the command with the copied config.

### Executed Figure Notebook

[View the executed notebook](notebooks/reproduce_selected_paper_figures.ipynb) to inspect the calculations and embedded outputs for Fig. 2b, Fig. 2c, Fig. 2e, Fig. 3b, Fig. 3c, Fig. 3d, Fig. 3e, Fig. 3f, and Fig. 3h. The notebook uses the archived 10-class model for Fig. 2c and does not retrain it. Plot outputs are retained inside the notebook; no separate figure files are created.

GitHub renders the saved outputs without requiring any setup. To execute the notebook locally, install the optional figure dependencies and launch Jupyter from the repository root:

```bash
python -m pip install -e ".[figures]"
python -m jupyter lab notebooks/reproduce_selected_paper_figures.ipynb
```

All inputs used by the notebook are bundled in `Figure_data/` or elsewhere in this repository, so a complete clone can reproduce every listed panel. No path configuration is needed when Jupyter is started from the repository root. Set `DAP_FIGURE_DATA` only to override the default `Figure_data/` location.

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
Reproduce the 100-class paper model using bundled PCA-space train/test matrices. See [Reproduce The Paper Results](#reproduce-the-paper-results) for the complete first-time workflow and expected results.

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
