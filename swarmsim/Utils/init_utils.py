import numpy as np
import pandas as pd
from pathlib import Path
import logging


def get_states(init_config: dict, num_samples: int, dim_samples: tuple or list) -> np.ndarray:
    """
    Loads or generates states for agents, obstacles or other simulation elements.
    Depending on the mode setting, the states are either loaded from file or generated randomly.
    """
    # Read the initialization mode
    mode = init_config.get('mode', "random").lower()

    # Initialization from file
    if mode == "file":
        states = _load_states_from_file(init_config)

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


def _load_states_from_file(init_config: dict) -> np.ndarray:
    """
    Helper method to load states from a CSV file.
    Validates the file and adjusts N and state_dim if necessary.
    """
    # Import initial conditions from file
    file_settings = init_config.get("file", {})
    file_path = file_settings.get("file_path", "")

    # Warning if file does not exist
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Initial conditions CSV file {file_path} not found.")
    states = pd.read_csv(file_path, header=None).values

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


def get_parameters(params_config: dict, num_samples: int) -> pd.DataFrame:
    """
    Loads or generates the parameters for the population.
    Depending on the mode setting, parameters are either loaded from file or generated online.
    """

    mode = params_config.get("mode")
    if mode == "file":
        settings = params_config.get("file")
        params = _load_parameters(settings, num_samples)
    elif mode == "generate":
        settings = params_config.get("generate")
        params = _generate_parameters(settings, num_samples)
    else:
        raise RuntimeError("Invalid parameter mode. Check the YAML config file.")
    return params

def _load_parameters(params_config: dict, num_samples: int) -> pd.DataFrame:
    """
    Helper method to load parameters from a CSV file.
    Verifies file existence and repeats rows if needed.
    """
    file_path = params_config.get("file_path", "")
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Parameters CSV file {file_path} not found.")
    params = pd.read_csv(file_path)
    if params.shape[0] < num_samples:
        logging.warning("Parameters CSV has fewer rows than N; repeating some rows to match agent count.")
        repeated_indices = np.random.choice(params.index, size=num_samples - params.shape[0], replace=True)
        params = pd.concat([params, params.iloc[repeated_indices]], ignore_index=True)
    return params


def _generate_parameters(params_config: dict, num_samples: int) -> pd.DataFrame:
    """
    Generate parameters for N samples based on the provided configuration.

    The configuration dictionary is expected to have a "generate" sub-dictionary where each
    parameter can be specified in one of two ways:

    1. Random generator specification: the value is a dictionary with keys such as:
       - "sampler": a string identifier (e.g., "uniform", "normal", "poisson", etc.)
       - "positional_args" (optional): list of positional arguments for the generator.
       - "args" (optional): dictionary of keyword arguments for the generator.
       - "shape" (optional): list or tuple specifying the shape of each sample
         (if omitted or empty, the parameter is assumed to be scalar).

       Example:
       ```yaml
       mu:
         sampler: normal
         positional_args: [0.0, 1.0]
         shape: []  # scalar
       ```

    2. Direct assignment: the value is specified directly (scalar, list, or nested list).
       Example:
       ```yaml
       K: 5
       A: [[1, 2], [3, 1]]
       ```

    Parameters:
        params_config : dict
            The configuration dictionary for parameters (typically corresponding to the "parameters" section).
            It should include a "generate" key, where each parameter is defined.
        num_samples : int
            The number of samples (agents) to generate parameters for.

    Returns:
        pd.DataFrame: A DataFrame with one column per parameter and one row per sample. Multi-dimensional
                      parameters are stored as arrays (one per sample).
    """

    params_df = pd.DataFrame()

    # Define an exhaustive mapping from sampler identifiers to numpy random generator functions.
    sampler_mapping = {
        "uniform": np.random.uniform,
        "normal": np.random.normal,
        "poisson": np.random.poisson,
        "exponential": np.random.exponential,
        "binomial": np.random.binomial,
        "gamma": np.random.gamma,
        "beta": np.random.beta,
        "chisquare": np.random.chisquare,
        "f": np.random.f,
        "geometric": np.random.geometric,
        "gumbel": np.random.gumbel,
        "hypergeometric": np.random.hypergeometric,
        "laplace": np.random.laplace,
        "logistic": np.random.logistic,
        "lognormal": np.random.lognormal,
        "logseries": np.random.logseries,
        "multinomial": np.random.multinomial,
        "multivariate_normal": np.random.multivariate_normal,
        "negative_binomial": np.random.negative_binomial,
        "noncentral_chisquare": np.random.noncentral_chisquare,
        "noncentral_f": np.random.noncentral_f,
        "pareto": np.random.pareto,
        "power": np.random.power,
        "rayleigh": np.random.rayleigh,
        "standard_cauchy": np.random.standard_cauchy,
        "standard_exponential": np.random.standard_exponential,
        "standard_gamma": np.random.standard_gamma,
        "standard_normal": np.random.standard_normal,
        "standard_t": np.random.standard_t,
        "triangular": np.random.triangular,
        "vonmises": np.random.vonmises,
        "wald": np.random.wald,
        "weibull": np.random.weibull,
        "zipf": np.random.zipf,
        # Extend further if needed.
    }

    # Process each parameter defined in the params_config.
    for par_name, settings in params_config.items():
        # Check if the parameter is specified directly (i.e. not as a dict).
        if not isinstance(settings, dict):
            # If it's a list, tuple, or similar, convert to a NumPy array for consistency.
            constant_value = np.array(settings) if isinstance(settings, (list, tuple)) else settings
            # Replicate this constant value across all N samples.
            # For scalars, we create an array; for arrays/matrices, we store each one as a separate element.
            if np.isscalar(constant_value):
                params_df[par_name] = np.full(num_samples, constant_value)
            else:
                params_df[par_name] = [constant_value for _ in range(num_samples)]
            continue

        # Otherwise, the parameter is specified by a dictionary and we use a random generator.
        sampler_key = settings.get("sampler", "uniform").lower()
        if sampler_key not in sampler_mapping:
            raise ValueError(f"Unsupported sampler: {sampler_key}")
        sampler = sampler_mapping[sampler_key]

        # Get positional arguments (if any).
        positional_args = settings.get("positional_args")
        # Get keyword arguments; copy to avoid mutation.
        args = settings.get("args", {}).copy()

        # Determine the output shape for one sample.
        shape = settings.get("shape", [])
        if isinstance(shape, list):
            shape = tuple(shape)

        # Check if the parameter should be homogeneous across all samples
        is_homogeneous = settings.get("homogeneous", False)

        if is_homogeneous:
            # Generate a single sample
            sample_size = shape if shape else ()
            args["size"] = sample_size
            if positional_args is not None:
                single_value = sampler(*positional_args, **args)
            else:
                single_value = sampler(**args)

            # Broadcast single sample to all agents
            if shape:
                values = [np.array(single_value) for _ in range(num_samples)]
                params_df[par_name] = values
            else:
                params_df[par_name] = np.full(num_samples, single_value)

        else:
            # Generate one sample per agent
            sample_size = (num_samples,) + shape if shape else num_samples
            args["size"] = sample_size
            if positional_args is not None:
                values = sampler(*positional_args, **args)
            else:
                values = sampler(**args)

            # Store multi-dimensional arrays as lists if needed
            if np.asarray(values).ndim > 1 and shape:
                params_df[par_name] = list(values)
            else:
                params_df[par_name] = values

    return params_df


def set_parameter(params: pd.Series, shape: tuple = ()) -> np.ndarray:
    """
    Set parameter values from a pandas Series to an array of shape (num_agents, *shape).

    For each element in `params`:
      - If the element is a scalar:
          * If `shape` is a square matrix (e.g. (n, n)), a diagonal matrix is created with the
            scalar on the diagonal.
          * Otherwise, an array of shape `shape` filled with the scalar is created.
      - If the element is array-like, it is converted to a NumPy array and its shape is checked.

    This function is optimized for speed by using vectorized operations:
      - For scalar parameters, it uses np.broadcast_to (or a vectorized diagonal assignment)
        so that no Python loop is needed.
      - For non-scalar parameters, it uses np.stack (which is implemented in C) to combine
        the individual arrays.

    Parameters
    ----------
    params : pd.Series
        A pandas Series in which each element is a parameter value (either a scalar or
        an array-like object).
    shape : tuple of int
        The desired shape for a single parameter value. For example, (state_dim,) for a vector
        or (state_dim, state_dim) for a square matrix.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (num_agents, *shape), where num_agents is the number of elements in `params`.
        If an element is scalar and a matrix is desired, a diagonal matrix is produced.

    Raises
    ------
    ValueError
        If any non-scalar parameter’s shape does not match the desired shape.
    """
    arr = params.to_numpy()
    num_agents = arr.shape[0]

    # Check if the first element is scalar (assuming homogeneous types in the Series)
    if np.isscalar(arr[0]):
        # Elements are scalars.
        if len(shape) == 2 and shape[0] == shape[1]:
            # When a square matrix is required, create an array of shape (num_agents, d, d),
            # then set the diagonal in a vectorized manner.
            d = shape[0]
            result = np.zeros((num_agents, d, d), dtype=np.result_type(arr))
            # Build diagonal indices.
            diag_indices = np.arange(d)
            # Reshape arr to (num_agents,1) so that each scalar fills the diagonal.
            result[:, diag_indices, diag_indices] = arr.reshape(num_agents, 1)
        else:
            # For non-square shapes, use broadcasting.
            # Reshape arr to (num_agents, 1, ..., 1) where the number of trailing 1's is len(shape)
            reshaped = arr.reshape((num_agents,) + (1,) * len(shape))
            result = np.broadcast_to(reshaped, (num_agents,) + shape)
    else:
        # Elements are array-like. Convert the first element and verify the shape.
        first_elem = np.array(arr[0])
        if first_elem.shape != shape:
            raise ValueError(f"Expected shape {shape} but got {first_elem.shape} for the first element.")
        # Stack all elements. np.stack calls an efficient C-level loop.
        result = np.stack(arr)

    return result
