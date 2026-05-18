import argparse
import logging
from pathlib import Path

import yaml

from . import pipeline


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


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def setup_logger(result_folder_path):
    log_path = Path(result_folder_path) / "process.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def main():
    args = parse_args()
    config = load_config(args.config_path)

    result_folder_path = config.get("result_folder_path")
    if not result_folder_path:
        raise ValueError("`result_folder_path` is required in the config file.")

    logger = setup_logger(result_folder_path)
    logger.info("Loaded configuration from %s", args.config_path)

    label_path = config.get("label_path")
    breed_list_text_path = config.get("breed_list_text_path")
    snp_csv_path = config.get("SNP_csv_path")
    pca_components = config.get("pca_components", 0.95)
    random_state = config.get("random_state", 42)
    test_size = config.get("test_size", 0.3)

    if args.reproduce:
        logger.info("Mode 6: Reproduce the 100-class model in the paper")
        pipeline.full_training_pipeline(
            result_folder_path=result_folder_path,
            breed_list_text_path="reproduce",
            pca_components=100,
            random_state=42,
            test_size=0.3,
        )
        return

    if args.train and args.inference:
        if not snp_csv_path:
            raise ValueError("Mode 4 requires `SNP_csv_path` in config.")
        if not label_path:
            raise ValueError("Mode 4 requires `label_path` in config.")
        logger.info("Mode 4: Train and test on the provided dataset")
        pipeline.full_training_pipeline(
            result_folder_path=result_folder_path,
            SNP_csv_path=snp_csv_path,
            label_path=label_path,
            breed_list_text_path=breed_list_text_path,
            pca_components=pca_components,
            random_state=random_state,
            test_size=test_size,
        )
        return

    if args.train:
        if not breed_list_text_path:
            raise ValueError("Mode 5 requires `breed_list_text_path` in config.")
        logger.info("Mode 5: Train on DAP data")
        pipeline.full_training_pipeline(
            result_folder_path=result_folder_path,
            breed_list_text_path=breed_list_text_path,
            pca_components=pca_components,
            random_state=random_state,
            test_size=test_size,
        )
        return

    if args.inference:
        if not snp_csv_path:
            raise ValueError("Modes 1/2/3 require `SNP_csv_path` in config.")

        if label_path is not None:
            logger.info("Mode 3: Train on DAP data and infer on provided samples with labels")
        elif breed_list_text_path is not None:
            logger.info("Mode 2: Train on DAP data and infer on provided samples with breed list")
        else:
            logger.info("Mode 1: Train on DAP data and infer on provided samples")

        input_args = {
            "result_folder_path": result_folder_path,
            "label_path": label_path,
            "breed_list_text_path": breed_list_text_path,
            "SNP_csv_path": snp_csv_path,
            "pca_components": pca_components,
            "random_state": random_state,
        }
        pipeline.train(**input_args)
        pipeline.inference(**input_args)


if __name__ == "__main__":
    main()

