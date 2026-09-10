import unittest
from pathlib import Path

import joblib
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


class ToyInputTests(unittest.TestCase):
    def test_modes_1_and_2_use_their_single_sample_inputs(self):
        expected_paths = {
            1: "./data/Toy_X_full_54143_single.csv",
            2: "./data/Toy_X_single.csv",
        }
        for mode, expected_path in expected_paths.items():
            config_path = REPO_ROOT / "configs" / f"config_mode_{mode}_template.yml"
            with config_path.open() as handle:
                config = yaml.safe_load(handle)
            self.assertEqual(config["SNP_csv_path"], expected_path)

    def test_full_sample_matches_pretrained_pca_schema(self):
        sample = pd.read_csv(REPO_ROOT / "data" / "Toy_X_full_54143_single.csv")
        pca = joblib.load(REPO_ROOT / "model" / "pca_model_WG_100.joblib")

        self.assertEqual(sample.shape, (1, 54144))
        self.assertEqual(sample.loc[0, "dog_id"], 109622)
        self.assertListEqual(
            list(sample.columns[1:]), [str(name) for name in pca.feature_names_in_]
        )

    def test_single_sample_input_matches_labeled_toy_schema(self):
        single = pd.read_csv(REPO_ROOT / "data" / "Toy_X_single.csv")
        labeled = pd.read_csv(REPO_ROOT / "data" / "Toy_X_snps.csv")

        self.assertEqual(single.shape, (1, labeled.shape[1]))
        self.assertEqual(single.loc[0, "dog_id"], 109622)
        self.assertListEqual(list(single.columns), list(labeled.columns))
        self.assertTrue(single.drop(columns="dog_id").notna().all(axis=None))

        reference = labeled.loc[labeled["dog_id"] == 109622].reset_index(drop=True)
        pd.testing.assert_frame_equal(single, reference)


if __name__ == "__main__":
    unittest.main()
