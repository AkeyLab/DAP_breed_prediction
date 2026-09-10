# Chromosome-wise DAP SNP Matrices

This directory contains the standardized, pruned DAP genotype matrices used
as training inputs by pipeline Modes 1, 2, 3, and 5. There is one Parquet file
for each of the 38 dog autosomes:

```text
X_SNP_ch1_pruned_v3_std.parquet
...
X_SNP_ch38_pruned_v3_std.parquet
```

Together, the files contain the 54,143 SNP features expected by the pipeline.
Rows are keyed by `dog_id`, and SNP columns follow the
`chr<chromosome>:<position>:<reference>:<alternate>` naming convention.
