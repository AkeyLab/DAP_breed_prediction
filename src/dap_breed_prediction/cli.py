import argparse

from .api import load_config, run_mode


def parse_args():
    parser = argparse.ArgumentParser(description="Dog breed prediction pipeline")

    parser.add_argument("-reproduce", "--reproduce", action="store_true", help="Reproduce the paper model (mode 6)")
    parser.add_argument("-train", "--train", action="store_true", help="Run training")
    parser.add_argument("-inference", "--inference", action="store_true", help="Run inference")
    parser.add_argument(
        "-config_path",
        "--config_path",
        type=str,
        required=True,
        help="Path to a YAML config file containing data paths and settings",
    )

    args = parser.parse_args()

    valid = (
        (args.reproduce and not args.train and not args.inference)
        or (args.train and not args.inference and not args.reproduce)
        or (args.train and args.inference and not args.reproduce)
        or (args.inference and not args.train and not args.reproduce)
    )
    if not valid:
        parser.error(
            "Invalid argument combination. Allowed:\n"
            "-inference (modes 1/2/3)\n"
            "-train -inference (mode 4)\n"
            "-train (mode 5)\n"
            "-reproduce (mode 6)"
        )

    return args


def select_mode(args, config):
    if args.reproduce:
        return 6
    if args.train and args.inference:
        return 4
    if args.train:
        return 5
    if config.get("label_path") is not None:
        return 3
    if config.get("breed_list_text_path") is not None:
        return 2
    return 1


def main():
    args = parse_args()
    config = load_config(args.config_path)
    mode = select_mode(args, config)
    run_mode(mode, config)


if __name__ == "__main__":
    main()
