# Figure Data

This directory contains the minimal additional inputs read by
`notebooks/reproduce_selected_paper_figures.ipynb`. Together with the PCA
matrices, labels, and 100-class model under `data/` and `model/`, these files
allow every notebook cell to run from a complete repository clone.

## Contents

| Directory | Contents | Panels |
| --- | --- | --- |
| `fig2c/` | Ten-class labels, archived 10-output model, and matching legacy 100-PC matrix | Fig. 2c |
| `fig2e/` | Per-class prediction summary from the reference 100-class run | Fig. 2e |
| `fig3/importance/per_breed/` | Fourteen breed-specific ranked SNP-importance tables | Fig. 3b |
| `fig3/importance/per_breed_with_frequency/` | Ranked importance tables with average allele frequencies | Fig. 3h |
| `fig3/overlap_and_similarity/` | Top/bottom overlap, Jaccard-index, and coverage tables | Fig. 3c, Fig. 3e, Fig. 3f |
| `fig3/reduced_snp_experiments/` | Eight reduced-SNP experiment summaries | Fig. 3d |

The notebook defaults to this directory. To use an alternative copy, set the
`DAP_FIGURE_DATA` environment variable before starting Jupyter.

These are the directly plotted analysis tables and model inputs. Raw
chromosome-level genotype files and the full PCA loading model are not
included because the notebook does not read them.
