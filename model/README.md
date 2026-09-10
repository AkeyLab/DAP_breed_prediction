# Archived 100-Class Model

This directory contains the fitted artifacts from the paper workflow:

- `regressor_model0_4-PCA100.pkl`: 100-output `MultiOutputRegressor` random forest operating on the first 100 principal components. Its SHA-256 digest is `8bb3ebf5c8900c9f065723d79d4f4c57207cb8a407f9b7af92aed7721a237e9f`, identical to the source file in `Dog_MAF0_4`.
- `pca_model_WG_100.joblib`: a 100-component copy of the source `pca_model_WG_1000.joblib`. It preserves the first 100 component vectors, PCA centering vector, explained variances, and all 54,143 SNP names in `feature_names_in_`.

The source PCA file was approximately 435 MB and exceeded GitHub's normal 100 MB per-file limit. The random forest accepts exactly 100 input features, so retaining components 101-1,000 would not change its inputs. A fixed test sample transformed by this trimmed PCA differed from the stored paper PCs by at most `5.03e-12`.

## Input Requirements

The PCA input must contain all 54,143 SNPs in `feature_names_in_` order and must already use the chromosome-wise standardization applied during paper preprocessing. The original fitted SNP scalers were not saved with the paper model. Do not apply this PCA directly to raw `0/1/2` genotype calls.

Mode 1 loads both artifacts for one-sample, 100-output inference and uses a configurable purity threshold that defaults to `0.7`. The bundled `data/Toy_X_full_54143_single.csv` demonstrates the required schema and preprocessing. Mode 5 separately trains a new random forest against the fixed PC matrices to reproduce the paper benchmark.

Only load pickle or Joblib files obtained from a trusted source, because deserialization can execute arbitrary code.
