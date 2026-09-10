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
            "prediction_model_path": "models/model.pkl",
            "model_metadata_path": "models/model_metadata.json",
            "pca_components": 0.9,
            "pure_threshold": 0.8,
            "random_state": 7,
            "test_size": 0.25,
            "include_unknown": False,
        }
        config.update(updates)
        return config

    @patch("dap_breed_prediction.api.pipeline.pretrained_inference")
    def test_mode_1_uses_bundled_pretrained_pipeline(self, pretrained_inference):
        config = self.config(
            prediction_model_path=None,
            model_metadata_path=None,
            label_path=None,
            breed_list_text_path=None,
        )
        result = run_mode(1, config, base_dir=self.base_dir, configure_logging=False)

        self.assertEqual(result, self.base_dir / "results/test")
        pretrained_inference.assert_called_once()
        kwargs = pretrained_inference.call_args.kwargs
        self.assertEqual(kwargs["SNP_csv_path"], str(self.base_dir / "data/snps.csv"))
        self.assertEqual(kwargs["pure_threshold"], 0.8)

    @patch("dap_breed_prediction.api.pipeline.pretrained_inference")
    def test_mode_1_uses_default_threshold(self, pretrained_inference):
        run_mode(
            1,
            {
                "result_folder_path": "results/test",
                "SNP_csv_path": "data/snps.csv",
            },
            base_dir=self.base_dir,
            configure_logging=False,
        )

        self.assertEqual(
            pretrained_inference.call_args.kwargs["pure_threshold"], 0.7
        )

    @patch("dap_breed_prediction.api.pipeline.pretrained_inference")
    def test_mode_1_treats_null_threshold_as_default(self, pretrained_inference):
        run_mode(
            1,
            {
                "result_folder_path": "results/test",
                "SNP_csv_path": "data/snps.csv",
                "pure_threshold": None,
            },
            base_dir=self.base_dir,
            configure_logging=False,
        )

        self.assertEqual(
            pretrained_inference.call_args.kwargs["pure_threshold"], 0.7
        )

    @patch("dap_breed_prediction.api.pipeline.inference")
    @patch("dap_breed_prediction.api.pipeline.train")
    def test_mode_2_retrains_then_predicts(self, train, inference):
        train.return_value = {
            "prediction_model_path": self.base_dir / "results/test/Model/model.pkl",
            "model_metadata_path": self.base_dir / "results/test/Model/model_metadata.json",
        }
        result = run_mode(2, self.config(), base_dir=self.base_dir, configure_logging=False)

        self.assertEqual(result, self.base_dir / "results/test")
        train.assert_called_once()
        self.assertEqual(train.call_args.kwargs["pure_threshold"], 0.8)
        inference.assert_called_once()
        kwargs = inference.call_args.kwargs
        self.assertFalse(kwargs["require_exact_features"])
        self.assertNotIn("label_path", kwargs)

    @patch("dap_breed_prediction.api.pipeline.inference")
    @patch("dap_breed_prediction.api.pipeline.train")
    def test_mode_2_treats_null_threshold_as_default(self, train, inference):
        train.return_value = {
            "prediction_model_path": self.base_dir / "results/test/Model/model.pkl",
            "model_metadata_path": self.base_dir / "results/test/Model/model_metadata.json",
        }
        run_mode(
            2,
            self.config(pure_threshold=None),
            base_dir=self.base_dir,
            configure_logging=False,
        )

        self.assertEqual(train.call_args.kwargs["pure_threshold"], 0.7)
        self.assertEqual(inference.call_args.kwargs["pure_threshold"], 0.7)

    @patch("dap_breed_prediction.api.pipeline.inference")
    @patch("dap_breed_prediction.api.pipeline.train")
    def test_mode_3_only_tests_supplied_model(self, train, inference):
        result = run_mode(3, self.config(), base_dir=self.base_dir, configure_logging=False)

        self.assertEqual(result, self.base_dir / "results/test")
        train.assert_not_called()
        inference.assert_called_once()
        kwargs = inference.call_args.kwargs
        self.assertTrue(kwargs["require_exact_features"])
        self.assertEqual(kwargs["pure_threshold"], 0.8)
        self.assertEqual(
            kwargs["prediction_model_path"], str(self.base_dir / "models/model.pkl")
        )
        self.assertEqual(kwargs["label_path"], str(self.base_dir / "data/labels.csv"))

    @patch("dap_breed_prediction.api.pipeline.full_training_pipeline")
    def test_mode_4_uses_only_user_data(self, full_training_pipeline):
        run_mode(4, self.config(), base_dir=self.base_dir, configure_logging=False)

        full_training_pipeline.assert_called_once()
        kwargs = full_training_pipeline.call_args.kwargs
        self.assertEqual(kwargs["SNP_csv_path"], str(self.base_dir / "data/snps.csv"))
        self.assertEqual(kwargs["label_path"], str(self.base_dir / "data/labels.csv"))

    @patch("dap_breed_prediction.api.pipeline.full_training_pipeline")
    def test_mode_5_reproduces_paper(self, full_training_pipeline):
        run_mode(
            5,
            {"result_folder_path": "results/reproduce"},
            base_dir=self.base_dir,
            configure_logging=False,
        )

        full_training_pipeline.assert_called_once()
        kwargs = full_training_pipeline.call_args.kwargs
        self.assertEqual(kwargs["breed_list_text_path"], "reproduce")
        self.assertEqual(kwargs["pca_components"], 100)
        self.assertEqual(kwargs["random_state"], 42)

    def test_mode_specific_required_fields(self):
        cases = {
            1: {"result_folder_path": "out"},
            2: {"result_folder_path": "out"},
            3: {
                "result_folder_path": "out",
                "SNP_csv_path": "x.csv",
                "prediction_model_path": "model.pkl",
            },
            4: {"result_folder_path": "out", "label_path": "y.csv"},
        }
        for mode, config in cases.items():
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, f"Mode {mode} requires"):
                    run_mode(mode, config, configure_logging=False)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "integer from 1 through 5"):
            run_mode(6, {"result_folder_path": "out"}, configure_logging=False)

    def test_yaml_config_and_relative_paths(self):
        config_path = self.base_dir / "config.yml"
        config_path.write_text(yaml.safe_dump(self.config()))
        loaded = load_config(config_path)
        self.assertEqual(loaded["random_state"], 7)

        with patch("dap_breed_prediction.api.pipeline.full_training_pipeline") as pipeline:
            result = run_mode(4, config_path, base_dir=self.base_dir, configure_logging=False)
        self.assertEqual(result, self.base_dir / "results/test")
        self.assertEqual(
            pipeline.call_args.kwargs["SNP_csv_path"],
            str(self.base_dir / "data/snps.csv"),
        )


if __name__ == "__main__":
    unittest.main()
