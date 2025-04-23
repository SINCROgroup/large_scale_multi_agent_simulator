import random
import numpy as np
import torch

import yaml
from pathlib import Path


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
# Verify that the configuration file exists
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, "r") as file:
        config_file = yaml.safe_load(file)

    return config_file
