import unittest
from argparse import Namespace

from dap_breed_prediction.cli import select_mode


class SelectModeTests(unittest.TestCase):
    def args(self, *, reproduce=False, train=False, inference=False):
        return Namespace(reproduce=reproduce, train=train, inference=inference)

    def test_explicit_modes(self):
        self.assertEqual(select_mode(self.args(reproduce=True), {}), 6)
        self.assertEqual(select_mode(self.args(train=True, inference=True), {}), 4)
        self.assertEqual(select_mode(self.args(train=True), {}), 5)

    def test_inference_modes_follow_config(self):
        inference = self.args(inference=True)
        self.assertEqual(select_mode(inference, {}), 1)
        self.assertEqual(select_mode(inference, {"breed_list_text_path": "breeds.txt"}), 2)
        self.assertEqual(select_mode(inference, {"label_path": "labels.csv"}), 3)


if __name__ == "__main__":
    unittest.main()
