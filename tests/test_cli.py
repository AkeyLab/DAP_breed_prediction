import unittest
from argparse import Namespace

from dap_breed_prediction.cli import select_mode


class SelectModeTests(unittest.TestCase):
    def args(self, *, reproduce=False, train=False, inference=False):
        return Namespace(reproduce=reproduce, train=train, inference=inference)

    def test_explicit_modes(self):
        self.assertEqual(select_mode(self.args(reproduce=True), {}), 5)
        self.assertEqual(select_mode(self.args(train=True, inference=True), {}), 2)
        self.assertEqual(select_mode(self.args(train=True), {}), 4)

    def test_inference_modes_follow_config(self):
        inference = self.args(inference=True)
        self.assertEqual(select_mode(inference, {}), 1)
        self.assertEqual(
            select_mode(inference, {"prediction_model_path": "model.pkl"}), 3
        )
        self.assertEqual(select_mode(inference, {"label_path": "labels.csv"}), 1)


if __name__ == "__main__":
    unittest.main()
