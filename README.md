# DAP Breed Prediction

Code and data package for dog breed prediction/admixture estimation used in the DAP paper workflows: "***An interpretable machine learning framework for dog breed inference and ancestry decomposition***".

## Start Here: Jupyter Tutorial

**New users should begin with the [standalone all-modes tutorial notebook](notebooks/tutorial_all_modes.ipynb).** It covers installation, input formats, safe inspection of the bundled Parquet files, Python API and CLI examples for every mode, and output inspection. The notebook has saved outputs, while model-training modes are disabled by default so users can choose explicitly which workflows to run.

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
│   ├── paper_14_breed_list.txt
│   ├── X_train_SNP_WG_prune_v3_1_std_pca_100.csv
│   ├── X_test_SNP_WG_prune_v3_1_std_pca_100.csv
│   ├── y_combined_100.csv
│   └── folder_of_54143_SNPs/            # Chromosome-wise DAP SNP matrices
├── notebooks/
│   ├── reproduce_selected_paper_figures.ipynb
│   └── tutorial_all_modes.ipynb
├── model/
│   ├── pca_model_WG_100.joblib          # Archived whole-genome PCA, first 100 PCs
│   ├── regressor_model0_4-PCA100.pkl     # Archived 100-class random forest
│   └── README.md                         # Model provenance and input requirements
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

The required PCA-space data and labels are included in this repository. Mode 6 does **not** read the chromosome-wise Parquet files used by Modes 1, 2, 3, and 5.

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

The separately bundled pretrained PCA and random-forest artifacts in `model/` are not used by this command; Mode 6 trains a new model to reproduce the training and evaluation procedure.

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

The `figures` extra pins scikit-learn to the version used for the saved notebook execution. All inputs used by the notebook are bundled in `Figure_data/` or elsewhere in this repository, so a complete clone can reproduce every listed panel. No path configuration is needed when Jupyter is started from the repository root. Set `DAP_FIGURE_DATA` only to override the default `Figure_data/` location.

### Archived Pretrained Artifacts

The source paper artifacts are bundled for inspection and advanced workflows:

- `model/regressor_model0_4-PCA100.pkl`: the archived 100-output random-forest model from `Dog_MAF0_4` (the repository copy is byte-for-byte identical to the source artifact).
- `model/pca_model_WG_100.joblib`: the first 100 components of the archived whole-genome PCA transformer. The source transformer contained 1,000 components and was approximately 435 MB, above GitHub's normal per-file limit. The random forest only consumes the first 100 components, so the unused 900 components were removed. The trimmed transformer was checked against the fixed test PCs with a maximum absolute difference of `5.03e-12`.

The archived PCA requires all 54,143 SNP values, already standardized and ordered according to its `feature_names_in_` attribute. The original per-chromosome standardization objects were not saved in `Dog_MAF0_4`; therefore these artifacts are not used as a plug-and-play raw-genotype inference path. Mode 1 instead trains a 100-output model using the standardized SNP columns available in the supplied CSV.

## Usage

### Python API

The package can be called directly from Python or a Jupyter notebook with a configuration dictionary:

```python
from pathlib import Path
from dap_breed_prediction import run_mode

config = {
    "result_folder_path": "./results/mode_4",
    "SNP_csv_path": "./data/Toy_X_snps.csv",
    "label_path": "./data/Toy_Y_labels.csv",
    "breed_list_text_path": "./data/Toy_a_short_breed_list.txt",
    "pca_components": 0.95,
    "random_state": 42,
    "test_size": 0.3,
}

result_directory = run_mode(4, config, base_dir=Path.cwd())
```

`config` can also be a path to one of the bundled YAML files. `base_dir` controls how relative configuration paths are resolved and defaults to the current working directory.

### CLI

Run from repository root:

```bash
python main.py <mode flags> -config_path <path_to_config.yml>
```

Or, if installed with `pip install -e .`:

```bash
dap-breed-predict <mode flags> -config_path <path_to_config.yml>
```

## Modes

### Mode 1: General 100-Class Prediction

**Designed for:** General inference on unlabeled dogs when their likely breed ancestry is not already narrowed to a small set.

**What it does:** Uses all 100 outputs in the bundled DAP label table (99 named breeds plus `Unknown`), trains a model using SNPs shared with the supplied genotype CSV, and predicts breed composition for the input dogs. Mode 1 trains this model for the SNPs available in the supplied CSV; it does not load the archived Mode 6 model.

- Flags: `-inference`
- Required config keys: `result_folder_path`, `SNP_csv_path`
- Optional config keys: `pca_components` (default `0.95`), `random_state` (default `42`)

### Mode 2: Predict With A Custom Breed Panel

**Designed for:** Predicting unlabeled dogs when the likely source breeds are known and the user wants a focused, custom set of output classes.

**What it does:** Reads one breed per line from `breed_list_text_path`, selects matching DAP reference dogs, trains on SNPs shared with the supplied genotype CSV, and predicts the input dogs using that targeted output panel. Set `include_unknown: true` to append an `Unknown` output; the default is `true` when this key is omitted.

The bundled Mode 2 template uses `data/paper_14_breed_list.txt` and `include_unknown: false`, producing an exact 14-output targeted model. These are the 14 breed outputs examined in the paper section **"Leveraging SNP importance scores to create small panels of informative variants"**:

```text
Australian Shepherd                 Beagle
Bernese Mountain Dog                Border Collie
Boston Terrier                      Cavalier King Charles Spaniel
Dachshund                           French Bulldog
German Shepherd Dog                 Golden Retriever
Great Dane                          Labrador Retriever
Pembroke Welsh Corgi                Poodle
```

The paper selected these outputs from its full 100-class model. Mode 2 instead retrains a focused model on the same 14-breed subset, so it is useful for targeted inference but is not an exact reproduction of the paper's fitted model or SNP-importance values. Use Mode 6 and the [executed figure notebook](notebooks/reproduce_selected_paper_figures.ipynb) for paper reproduction.

- Flags: `-inference`
- Required config keys: `result_folder_path`, `SNP_csv_path`, `breed_list_text_path`
- Optional config keys: `include_unknown` (default `true`), `pca_components` (default `0.95`), `random_state` (default `42`)

### Mode 3: Evaluate Predictions Against Known Labels

**Designed for:** Benchmarking performance on a labeled validation dataset rather than only generating predictions.

**What it does:** Trains from DAP reference dogs, predicts the supplied genotype CSV, compares predictions with `label_path`, and reports strict/loose accuracy plus per-class and confusion-map results. If a breed list is provided, it defines the classes; otherwise classes are inferred from the labels.

- Flags: `-inference`
- Required config keys: `result_folder_path`, `SNP_csv_path`, `label_path`
- Optional config keys: `breed_list_text_path`, `include_unknown` (default `true`), `pca_components` (default `0.95`), `random_state` (default `42`)

### Mode 4: Train And Test On Your Own Dataset

**Designed for:** Developing or testing a model entirely from a user-provided labeled dataset, without using the bundled DAP reference genotypes.

**What it does:** Splits the supplied genotype CSV and labels into training and test sets, optionally applies PCA, trains a model, and evaluates held-out samples. This is the fastest introductory workflow with the bundled toy files.

- Flags: `-train -inference`
- Required config keys: `result_folder_path`, `SNP_csv_path`, `label_path`
- Optional config keys: `breed_list_text_path`, `pca_components` (default `0.95`), `random_state` (default `42`), `test_size` (default `0.3`)

### Mode 5: Train A Reusable Model From The Full DAP Panel

**Designed for:** Building a new reference model for a chosen breed panel when immediate prediction of an external CSV is not required.

**What it does:** Loads all 38 bundled chromosome Parquet files, selects DAP dogs matching the requested breeds, performs a train/test split, applies PCA when needed, and writes the trained model and evaluation outputs. This is the most memory-intensive mode because it can read all 54,143 SNPs.

- Flags: `-train`
- Required config keys: `result_folder_path`, `breed_list_text_path`
- Optional config keys: `include_unknown` (default `true`), `pca_components` (default `0.95`), `random_state` (default `42`), `test_size` (default `0.3`)

### Mode 6: Reproduce The Paper Benchmark

**Designed for:** Reproducing the paper's fixed 100-class breed-prediction experiment rather than analyzing new samples.

**What it does:** Uses the bundled 100-PC training and test matrices, trains a new random forest with the paper settings, selects the pure/mixed threshold on the training set, and evaluates the fixed test set. It does not read the chromosome Parquet files. See [Reproduce The Paper Results](#reproduce-the-paper-results) for expected metrics.

- Flags: `-reproduce`
- Required config key: `result_folder_path`

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
  - SNP values must use the same standardization as the corresponding bundled DAP reference columns.
- `label_path` (CSV):
  - Must include `dog_id` and `label`.
  - Label format:
    - Pure: `BreedName`
    - Mixed: `BreedA / BreedB`
- `breed_list_text_path` (TXT):
  - One breed name per line.
- `include_unknown` (boolean; Modes 2, 3, and 5):
  - `true` appends the `Unknown` output to the classes in the breed list or labels.
  - `false` keeps only the named classes; the bundled Mode 2 template uses this setting for its 14-breed model.

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

## Bundled DAP Data for Modes 1/2/3/5

Modes `1/2/3/5` require chromosome-wise DAP SNP parquet files at:

```text
data/folder_of_54143_SNPs/X_SNP_ch*_pruned_v3_std.parquet
```

The 38 required Parquet files, one for each autosome, are bundled at that path. A complete repository clone therefore contains the DAP training matrices expected by these modes.

## Quick Smoke Test (Mode 4 With Toy Data)

```bash
python main.py -train -inference -config_path configs/config_mode_4_template.yml
```
