import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dap_breed_prediction import pipeline


class ModelContractTests(unittest.TestCase):
    def test_mode_1_requires_exactly_one_sample(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "two_samples.csv"
            pd.DataFrame(
                {
                    "dog_id": ["dog-a", "dog-b"],
                    "chr1:test:A:G": [0.0, 1.0],
                }
            ).to_csv(input_path, index=False)

            with self.assertRaisesRegex(ValueError, "exactly one sample row"):
                pipeline.pretrained_inference(
                    result_folder_path=root / "results",
                    SNP_csv_path=input_path,
                )

    def test_model_snp_schema_accepts_reordering(self):
        pipeline._validate_exact_snp_list(
            ["chr2:test:C:T", "chr1:test:A:G"],
            ["chr1:test:A:G", "chr2:test:C:T"],
        )

    def test_model_snp_schema_rejects_missing_and_extra_features(self):
        with self.assertRaisesRegex(
            ValueError, "Expected 2, received 2; missing 1, extra 1"
        ):
            pipeline._validate_exact_snp_list(
                ["chr1:test:A:G", "chr3:test:G:A"],
                ["chr1:test:A:G", "chr2:test:C:T"],
            )


if __name__ == "__main__":
    unittest.main()
