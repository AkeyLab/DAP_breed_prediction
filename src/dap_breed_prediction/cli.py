import argparse

from .api import run_mode


def parse_args():
    parser = argparse.ArgumentParser(description="Dog breed prediction pipeline")

    parser.add_argument(
        "-mode",
        "--mode",
        type=int,
        choices=range(1, 6),
        required=True,
        metavar="{1,2,3,4,5}",
        help="Pipeline mode to run",
    )
    parser.add_argument(
        "-config_path",
        "--config_path",
        type=str,
        required=True,
        help="Path to a YAML config file containing data paths and settings",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_mode(args.mode, args.config_path)


if __name__ == "__main__":
    main()
