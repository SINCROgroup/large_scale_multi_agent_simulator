"""
Core simulation utilities for configuration management and reproducibility.

This module provides fundamental utilities for simulation setup, including
configuration file loading, random seed management, and reproducible
simulation execution across different frameworks.
"""

import random
import numpy as np
import torch

import yaml
from pathlib import Path


def set_global_seed(seed):
    """
    Set reproducible random seeds across all major frameworks and libraries.

    This function ensures reproducible results by setting seeds for Python's
    built-in random module, NumPy, and PyTorch (both CPU and GPU). This is
    essential for scientific reproducibility and debugging simulations.

    Parameters
    ----------
    seed : int
        Random seed value to use across all frameworks.


    Applications
    ------------
    - **Scientific Reproducibility**: Ensure consistent results across runs
    - **Debugging**: Reproduce exact problematic scenarios
    - **Benchmarking**: Fair comparison between algorithms

    Notes
    -----
    - Should be called before any random operations
    - PyTorch CUDA seeding affects all available GPU devices
    - Does not affect system-level randomness (e.g., thread scheduling)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """
    Load and validate YAML configuration files for simulation setup.

    This function provides robust loading of YAML configuration files with
    proper error handling and validation. It's the standard method for loading
    simulation parameters, component configurations, and experimental settings.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file to load.

    Returns
    -------
    dict
        Parsed configuration dictionary with nested structure preserved.

    Raises
    ------
    FileNotFoundError
        If the specified configuration file does not exist.
    yaml.YAMLError
        If the file contains invalid YAML syntax.

    Notes
    -----
    - Uses PyYAML's safe_load for security
    - Preserves nested dictionary structure
    - Supports all standard YAML data types
    - Path can be absolute or relative to working directory
    - Configuration files should use .yaml or .yml extension
    """
    
    # Verify that the configuration file exists
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, "r") as file:
        config_file = yaml.safe_load(file)

    return config_file
