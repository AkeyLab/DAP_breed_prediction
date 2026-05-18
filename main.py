from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dap_breed_prediction.cli import main  # noqa: E402


if __name__ == "__main__":
    main()

