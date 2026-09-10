import unittest
from pathlib import Path

from dap_breed_prediction import pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]


class BreedClassTests(unittest.TestCase):
    def test_default_inference_uses_all_100_classes(self):
        classes = pipeline.get_all_breed_classes()

        self.assertEqual(len(classes), 100)
        self.assertEqual(len(set(classes)), 100)
        self.assertIn("Unknown", classes)

    def test_paper_breed_file_matches_selected_outputs(self):
        breed_file = REPO_ROOT / "data" / "paper_14_breed_list.txt"
        breeds = [line.strip() for line in breed_file.read_text().splitlines() if line.strip()]

        self.assertEqual(breeds, pipeline.PAPER_14_BREEDS)
        self.assertEqual(len(breeds), 14)
        self.assertNotIn("Unknown", breeds)

    def test_unknown_class_can_be_enabled_or_disabled(self):
        without_unknown = pipeline.configure_unknown_class(pipeline.PAPER_14_BREEDS, False)
        with_unknown = pipeline.configure_unknown_class(pipeline.PAPER_14_BREEDS, True)

        self.assertEqual(len(without_unknown), 14)
        self.assertNotIn("Unknown", without_unknown)
        self.assertEqual(len(with_unknown), 15)
        self.assertEqual(with_unknown[-1], "Unknown")


if __name__ == "__main__":
    unittest.main()
