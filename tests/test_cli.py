import unittest
from argparse import Namespace
from unittest.mock import patch

from dap_breed_prediction.cli import main, parse_args


class CliTests(unittest.TestCase):
    def test_short_options_select_mode_and_config(self):
        with patch(
            "sys.argv",
            ["dap-breed-predict", "-mode", "3", "-config_path", "config.yml"],
        ):
            args = parse_args()

        self.assertEqual(args.mode, 3)
        self.assertEqual(args.config_path, "config.yml")

    def test_long_options_select_mode_and_config(self):
        with patch(
            "sys.argv",
            ["dap-breed-predict", "--mode", "5", "--config_path", "config.yml"],
        ):
            args = parse_args()

        self.assertEqual(args.mode, 5)
        self.assertEqual(args.config_path, "config.yml")

    def test_mode_outside_supported_range_is_rejected(self):
        with patch(
            "sys.argv",
            ["dap-breed-predict", "-mode", "6", "-config_path", "config.yml"],
        ):
            with self.assertRaises(SystemExit):
                parse_args()

    @patch("dap_breed_prediction.cli.run_mode")
    @patch("dap_breed_prediction.cli.parse_args")
    def test_main_passes_explicit_mode_to_python_api(self, parse_args_mock, run_mode):
        parse_args_mock.return_value = Namespace(mode=2, config_path="config.yml")

        main()

        run_mode.assert_called_once_with(2, "config.yml")


if __name__ == "__main__":
    unittest.main()
