import json
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional

from src.paths import CONFIG_FILE


def get_logger(name: Optional[str] = None) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(name)
    return logger


def load_config(filepath: Path = CONFIG_FILE) -> Namespace:
    with open(filepath, "r") as config:
        return Namespace(**json.load(config))


def save_config(config: Namespace, filepath: Path) -> None:
    with open(filepath, "w") as file:
        json.dump(vars(config), file, indent=4)  # type: ignore


def nearest_divisible(x: int, y: int) -> int:
    return x - (x % y) if x % y <= y / 2 else x + (y - x % y)
