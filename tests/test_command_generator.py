import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from dap_breed_prediction import command_generator


class CommandGeneratorTests(unittest.TestCase):
    def test_mode_4_xgboost_config_and_command(self):
        parser = command_generator.build_parser()
        args = parser.parse_args(
            [
                "--mode",
                "4",
                "--config-path",
                "configs/xgboost_mode4.yml",
                "--training-model-name",
                "xgboost",
                "--xgboost-device",
                "cuda",
                "--result-folder-path",
                "./results/xgboost_mode4",
            ]
        )

        config = command_generator.build_config(args.mode, command_generator._selected_overrides(args))
        command = command_generator.build_run_command(args.mode, args.config_path)

        self.assertEqual(config["training_model_name"], "xgboost")
        self.assertEqual(config["xgboost_device"], "cuda")
        self.assertEqual(config["result_folder_path"], "./results/xgboost_mode4")
        self.assertEqual(
            command,
            "python main.py -mode 4 -config_path configs/xgboost_mode4.yml",
        )

    def test_can_clear_optional_breed_list(self):
        parser = command_generator.build_parser()
        args = parser.parse_args(["--mode", "4", "--breed-list-text-path", ""])

        config = command_generator.build_config(args.mode, command_generator._selected_overrides(args))

        self.assertIsNone(config["breed_list_text_path"])

    def test_mode_rejects_unsupported_override(self):
        with self.assertRaisesRegex(ValueError, "Mode 5 does not use"):
            command_generator.build_config(5, {"training_model_name": "xgboost"})

    def test_generate_writes_yaml_and_returns_command(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "mode_1.yml"
            parser = command_generator.build_parser()
            args = parser.parse_args(
                [
                    "--mode",
                    "1",
                    "--config-path",
                    str(config_path),
                    "--pretrained-model-name",
                    "ridge",
                ]
            )

            written_path, command = command_generator.generate(args)

            self.assertEqual(written_path, config_path)
            with config_path.open() as handle:
                config = yaml.safe_load(handle)
            self.assertEqual(config["pretrained_model_name"], "ridge")
            self.assertIn(f"-config_path {config_path}", command)

    def test_main_prints_run_this_command(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "mode_5.yml"
            argv = ["--mode", "5", "--config-path", str(config_path)]
            with patch("builtins.print") as print_mock:
                command_generator.main(argv)

            printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list)
            self.assertIn("Run this command:", printed)
            self.assertIn(f"python main.py -mode 5 -config_path {config_path}", printed)


if __name__ == "__main__":
    unittest.main()
