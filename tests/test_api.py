import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from dap_breed_prediction.api import load_config, run_mode


class RunModeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def config(self, **updates):
        config = {
            "result_folder_path": "results/test",
            "SNP_csv_path": "data/snps.csv",
            "label_path": "data/labels.csv",
            "breed_list_text_path": "data/breeds.txt",
            "pca_components": 0.9,
            "random_state": 7,
            "test_size": 0.25,
            "include_unknown": False,
        }
        config.update(updates)
        return config

    @patch("dap_breed_prediction.api.pipeline.inference")
    @patch("dap_breed_prediction.api.pipeline.train")
    def test_modes_1_to_3_dispatch_train_then_inference(self, train, inference):
        for mode in (1, 2, 3):
            with self.subTest(mode=mode):
                train.reset_mock()
                inference.reset_mock()
                result = run_mode(
                    mode,
                    self.config(),
                    base_dir=self.base_dir,
                    configure_logging=False,
                )
                self.assertEqual(result, self.base_dir / "results/test")
                train.assert_called_once()
                inference.assert_called_once_with(**train.call_args.kwargs)
                kwargs = train.call_args.kwargs
                if mode == 1:
                    self.assertNotIn("label_path", kwargs)
                    self.assertNotIn("breed_list_text_path", kwargs)
                    self.assertTrue(kwargs["include_unknown"])
                    self.assertEqual(kwargs["pure_threshold"], 0.7)
                elif mode == 2:
                    self.assertNotIn("label_path", kwargs)
                    self.assertEqual(kwargs["breed_list_text_path"], str(self.base_dir / "data/breeds.txt"))
                    self.assertFalse(kwargs["include_unknown"])
                    self.assertEqual(kwargs["pure_threshold"], 0.7)
                else:
                    self.assertEqual(kwargs["label_path"], str(self.base_dir / "data/labels.csv"))
                    self.assertEqual(kwargs["breed_list_text_path"], str(self.base_dir / "data/breeds.txt"))
                    self.assertFalse(kwargs["include_unknown"])
                    self.assertIsNone(kwargs["pure_threshold"])

    @patch("dap_breed_prediction.api.pipeline.inference")
    @patch("dap_breed_prediction.api.pipeline.train")
    def test_modes_1_and_2_accept_configured_threshold(self, train, inference):
        for mode in (1, 2):
            with self.subTest(mode=mode):
                train.reset_mock()
                inference.reset_mock()
                run_mode(
                    mode,
                    self.config(pure_threshold=0.8),
                    base_dir=self.base_dir,
                    configure_logging=False,
                )
                self.assertEqual(train.call_args.kwargs["pure_threshold"], 0.8)
                self.assertEqual(inference.call_args.kwargs["pure_threshold"], 0.8)

    @patch("dap_breed_prediction.api.pipeline.full_training_pipeline")
    def test_modes_4_to_6_dispatch_full_pipeline(self, full_training_pipeline):
        expected = {
            4: {"SNP_csv_path", "label_path"},
            5: {"breed_list_text_path"},
            6: {"breed_list_text_path"},
        }
        for mode in (4, 5, 6):
            with self.subTest(mode=mode):
                full_training_pipeline.reset_mock()
                run_mode(
                    mode,
                    self.config(),
                    base_dir=self.base_dir,
                    configure_logging=False,
                )
                full_training_pipeline.assert_called_once()
                kwargs = full_training_pipeline.call_args.kwargs
                self.assertTrue(expected[mode].issubset(kwargs))
                if mode == 4:
                    self.assertNotIn("include_unknown", kwargs)
                if mode == 5:
                    self.assertFalse(kwargs["include_unknown"])
                if mode == 6:
                    self.assertEqual(kwargs["breed_list_text_path"], "reproduce")
                    self.assertEqual(kwargs["pca_components"], 100)
                    self.assertEqual(kwargs["random_state"], 42)

    def test_mode_specific_required_fields(self):
        cases = {
            1: {"result_folder_path": "out"},
            2: {"result_folder_path": "out", "SNP_csv_path": "x.csv"},
            3: {"result_folder_path": "out", "SNP_csv_path": "x.csv"},
            4: {"result_folder_path": "out", "label_path": "y.csv"},
            5: {"result_folder_path": "out"},
        }
        for mode, config in cases.items():
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, f"Mode {mode} requires"):
                    run_mode(mode, config, configure_logging=False)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "integer from 1 through 6"):
            run_mode(7, {"result_folder_path": "out"}, configure_logging=False)

    def test_yaml_config_and_relative_paths(self):
        config_path = self.base_dir / "config.yml"
        config_path.write_text(yaml.safe_dump(self.config()))
        loaded = load_config(config_path)
        self.assertEqual(loaded["random_state"], 7)

        with patch("dap_breed_prediction.api.pipeline.full_training_pipeline") as pipeline:
            result = run_mode(
                4,
                config_path,
                base_dir=self.base_dir,
                configure_logging=False,
            )
        self.assertEqual(result, self.base_dir / "results/test")
        self.assertEqual(
            pipeline.call_args.kwargs["SNP_csv_path"],
            str(self.base_dir / "data/snps.csv"),
        )


if __name__ == "__main__":
    unittest.main()
