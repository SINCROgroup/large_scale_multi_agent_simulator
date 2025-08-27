"""
Agent initialization utilities for multi-agent simulations.

This module provides flexible and robust utilities for initializing agent states
in multi-agent simulations. It supports both random initialization with various
spatial distributions and deterministic initialization from data files.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def get_states(init_config: dict, num_samples: int, dim_samples: tuple or list) -> np.ndarray:
    """
    Generate or load initial states for agents in multi-agent simulations.

    This function provides flexible initialization of agent states, supporting both
    random generation with configurable spatial distributions and deterministic
    loading from data files. It's designed to handle various initialization
    scenarios for different types of multi-agent systems.

    Parameters
    ----------
    init_config : dict
        Configuration dictionary specifying initialization mode and parameters.
    num_samples : int
        Number of agents to initialize.
    dim_samples : tuple or list
        Dimensions of the state vector for each agent.

    Returns
    -------
    np.ndarray
        Initial states array with shape (num_samples, dim_samples).

    Configuration Structure
    -----------------------
    The init_config dictionary supports two main modes:

    **Random Mode**:
    
    .. code-block:: yaml

        mode: "random"
        random:
            shape: "box"  # or "circle"
            box:
                lower_bounds: [-10, -10, 0, 0]
                upper_bounds: [10, 10, 0, 0]
            # OR
            circle:
                min_radius: 0
                max_radius: 5
                lower_bounds_other_states: [0, 0]
                upper_bounds_other_states: [0, 0]

    **File Mode**:
    
    .. code-block:: yaml

        mode: "file"
        file:
            file_path: "initial_conditions.csv"  # or .npz

    Initialization Modes
    --------------------
    **Box Distribution**: Uniform random initialization within hyperrectangular bounds
    **Circle Distribution**: Uniform distribution within circular region (first 2D), uniform for other dimensions
    **File Loading**: Direct loading from CSV or NPZ files with validation

    Examples
    --------
    Box initialization for 2D agents with velocities:

    .. code-block:: python

        from swarmsim.Utils import get_states

        box_config = {
            'mode': 'random',
            'random': {
                'shape': 'box',
                'box': {
                    'lower_bounds': [-50, -50, -2, -2],  # [x_min, y_min, vx_min, vy_min]
                    'upper_bounds': [50, 50, 2, 2]       # [x_max, y_max, vx_max, vy_max]
                }
            }
        }
        
        states = get_states(box_config, num_samples=100, dim_samples=4)
        print(f"Initialized {states.shape[0]} agents with {states.shape[1]}D states")

    Circular initialization for spatial clustering:

    .. code-block:: python

        circle_config = {
            'mode': 'random',
            'random': {
                'shape': 'circle',
                'circle': {
                    'min_radius': 2.0,
                    'max_radius': 10.0,
                    'lower_bounds_other_states': [-1, -1],  # For velocity components
                    'upper_bounds_other_states': [1, 1]
                }
            }
        }
        
        states = get_states(circle_config, num_samples=50, dim_samples=4)

    File-based initialization from CSV:

    .. code-block:: python

        file_config = {
            'mode': 'file',
            'file': {
                'file_path': 'predefined_formation.csv'
            }
        }
        
        states = get_states(file_config, num_samples=20, dim_samples=6)

    
    Error Handling
    --------------
    The function provides comprehensive error checking:
    
    - **File Validation**: Checks file existence and format compatibility
    - **Dimension Validation**: Ensures state dimensions match expectations
    - **Agent Count Validation**: Verifies number of agents in file data
    - **Configuration Validation**: Validates all required parameters

    Notes
    -----
    - Box initialization supports arbitrary dimensionality
    - Circle initialization applies to first 2 dimensions only
    - File formats must match expected agent count and state dimensions
    - Random number generation uses numpy's global random state
    - All bounds are inclusive for uniform distributions
    """
    # Read the initialization mode
    mode = init_config.get('mode', "random").lower()

    # Initialization from file
    if mode == "file":
        states = _load_states_from_file(init_config, num_samples, dim_samples)

    # Random initialization
    elif mode == "random":
        random_settings = init_config.get("random", {})
        shape = random_settings.get("shape", "box")
        # Agents initialized in a box
        if shape == "box":
            box_settings = random_settings.get("box")
            states = _generate_random_states_box(box_settings, num_samples, dim_samples)
        # Agents initialized in a circle
        elif shape == "circle":
            circle_settings = random_settings.get("circle")
            states = _generate_random_states_circle(circle_settings, num_samples, dim_samples)
        else:
            raise RuntimeError(f"Unknown initialization' shape: {shape} (choose between 'box' and 'circle')")
    else:
        raise RuntimeError("Invalid initialization mode. Check the YAML config file (choose between 'random' and 'file').")

    return states


def _load_states_from_file(init_config: dict, num_samples: int, dim_samples: int) -> np.ndarray:
    """
    Helper method to load states from CSV or NPZ files.
    Validates the file and adjusts num_samples and dim_samples if necessary.
    """
    file_settings = init_config.get("file", {})
    file_path = Path(file_settings.get("file_path", ""))

    if not file_path.exists():
        raise FileNotFoundError(f"Initial conditions file '{file_path}' not found.")

    # Load from CSV file
    if file_path.suffix == ".csv":
        states = pd.read_csv(file_path, header=None).values

    # Load from NPZ file
    elif file_path.suffix == ".npz":
        npz_data = np.load(file_path)
        if 'states' not in npz_data:
            raise KeyError("The .npz file does not contain a 'states' array.")
        states = npz_data['states']

    else:
        raise ValueError(f"Unsupported file type '{file_path.suffix}'. Only '.csv' and '.npz' are supported.")

    # Validate the number of agents (rows)
    if states.shape[0] != num_samples:
        raise ValueError(
            f"Inconsistent number of agents in the file '{file_path.name}': "
            f"{states.shape[0]} vs expected {num_samples}."
        )

    # Validate the number of states (columns)
    if states.shape[1] != dim_samples:
        raise ValueError(
            f"Inconsistent number of states per agent in the file '{file_path.name}': "
            f"{states.shape[1]} vs expected {dim_samples}."
        )

    return states



def _generate_random_states_box(box_settings: dict, num_samples: int, dim_samples: tuple or list) -> np.ndarray:
    """
    Helper method to generate random states within a hyper-rectangle (box) defined by lower and upper bounds.
    """
    lower_bounds = box_settings.get("lower_bounds")
    upper_bounds = box_settings.get("upper_bounds")

    if lower_bounds is None or upper_bounds is None:
        raise ValueError("Missing lower_bounds or upper_bounds in the box_settings.")

    if len(lower_bounds) != dim_samples or len(upper_bounds) != dim_samples:
        raise ValueError("The length of lower_bounds and upper_bounds must be the same as state_dim.")

    states = np.random.uniform(lower_bounds, upper_bounds, [num_samples, dim_samples])
    return states


def _generate_random_states_circle(circle_settings, num_samples: int, dim_samples: tuple or list) -> np.ndarray:
    """
    Helper method to generate random states in a circular distribution for the first two dimensions.
    Any additional state dimensions are populated using other states bounds if provided, or default to 0.
    """

    max_radius = circle_settings.get("max_radius")
    min_radius = circle_settings.get("min_radius", 0)

    if max_radius is None or min_radius is None:
        raise ValueError("Missing max_radius or min_radius in the circle_settings.")

    if max_radius < min_radius:
        raise ValueError("max_radius must be greater than or equal to min_radius.")

    theta = np.random.uniform(0, 2 * np.pi, num_samples)
    radii_squared = np.random.uniform(min_radius ** 2, max_radius ** 2, num_samples)
    radii = np.sqrt(radii_squared)

    states = np.zeros((num_samples, dim_samples))
    states[:, :2] = np.column_stack((radii * np.cos(theta), radii * np.sin(theta)))

    # For dimensions beyond two, initialize other states uniformly
    if dim_samples > 2:
        default_bounds = [0.0] * int(dim_samples - 2)

        lower_bounds_other_states = circle_settings.get("lower_bounds_other_states", default_bounds)
        upper_bounds_other_states = circle_settings.get("upper_bounds_other_states", default_bounds)

        if lower_bounds_other_states is None or upper_bounds_other_states is None:
            raise ValueError("Missing lower_bounds or upper_bounds for other states in the circle_settings.")

        if len(lower_bounds_other_states) != (dim_samples - 2) or len(upper_bounds_other_states) != (
                dim_samples - 2):
            raise ValueError("The length of lower_bounds and upper_bounds must be the same as (state_dim - 2).")

        states[:, 2:] = np.random.uniform(lower_bounds_other_states,
                                          upper_bounds_other_states,
                                          size=(num_samples, dim_samples - 2))

    return states


