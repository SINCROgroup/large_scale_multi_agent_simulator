import numpy as np
import pandas as pd
from pathlib import Path
import logging


def get_parameters(params_config: dict, params_shapes: dict[str, tuple], num_samples: int) -> dict[str, np.ndarray]:
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

    if params_shapes is None:
        raise ValueError("params_shapes must be defined before initializing parameters.")

    missing = [k for k in params_shapes if k not in params]
    if missing:
        raise KeyError(f"Missing parameter(s) in params: {missing}")

    return {
        key: _reshape_parameter(params[key], shape)
        for key, shape in params_shapes.items()
    }


def _load_parameters(params_config: dict, num_samples: int) -> dict[str, np.ndarray]:
    """
    Helper method to load parameters from CSV or NPZ file.
    Verifies file existence, repeats rows if too few, and truncates if too many.
    Returns a dictionary of parameters with numpy arrays.
    """
    file_path = Path(params_config.get("file_path", ""))

    if not file_path.exists():
        raise FileNotFoundError(f"Parameters file '{file_path}' not found.")

    # Load from CSV file using pandas for convenience
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
        params_dict = {col: df[col].to_numpy() for col in df.columns}

    # Load from NPZ file
    elif file_path.suffix == ".npz":
        npz_data = np.load(file_path)
        params_dict = {key: npz_data[key] for key in npz_data.files}

    else:
        raise ValueError(f"Unsupported file type '{file_path.suffix}'. Only '.csv' and '.npz' are supported.")

    current_num_samples = next(iter(params_dict.values())).shape[0]

    # Repeat rows if not enough
    if current_num_samples < num_samples:
        logging.warning("Parameters file has fewer rows than N; repeating some rows to match agent count.")
        repeat_indices = np.random.choice(current_num_samples, size=num_samples - current_num_samples, replace=True)
        for key in params_dict:
            params_dict[key] = np.concatenate([params_dict[key], params_dict[key][repeat_indices]])

    # Truncate if too many rows
    elif current_num_samples > num_samples:
        logging.warning("Parameters file has more rows than N; truncating to match agent count.")
        for key in params_dict:
            params_dict[key] = params_dict[key][:num_samples]

    return params_dict


def _generate_parameters(params_config: dict, num_samples: int) -> dict[str, np.ndarray]:
    """
    Generate parameters for N samples based on the provided configuration.
    Returns a dictionary with one key per parameter, each containing a NumPy array.
    """
    params_dict = {}

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
    }

    for par_name, settings in params_config.items():
        if not isinstance(settings, dict):
            constant_value = np.array(settings) if isinstance(settings, (list, tuple)) else settings
            if np.isscalar(constant_value):
                params_dict[par_name] = np.full(num_samples, constant_value)
            else:
                stacked_array = np.stack([constant_value] * num_samples)
                params_dict[par_name] = stacked_array
            continue

        sampler_key = settings.get("sampler", "uniform").lower()
        if sampler_key not in sampler_mapping:
            raise ValueError(f"Unsupported sampler: {sampler_key}")
        sampler = sampler_mapping[sampler_key]

        positional_args = settings.get("positional_args")
        args = settings.get("args", {}).copy()

        shape = settings.get("shape", [])
        if isinstance(shape, list):
            shape = tuple(shape)

        is_homogeneous = settings.get("homogeneous", False)

        if is_homogeneous:
            args["size"] = shape if shape else ()
            if positional_args is not None:
                single_value = sampler(*positional_args, **args)
            else:
                single_value = sampler(**args)

            if shape:
                stacked_array = np.stack([np.array(single_value) for _ in range(num_samples)])
                params_dict[par_name] = stacked_array
            else:
                params_dict[par_name] = np.full(num_samples, single_value)
        else:
            args["size"] = (num_samples,) + shape if shape else num_samples
            if positional_args is not None:
                values = sampler(*positional_args, **args)
            else:
                values = sampler(**args)
            params_dict[par_name] = np.array(values)

    return params_dict


def _reshape_parameter(param_array: np.ndarray, shape: tuple = ()) -> np.ndarray:
    """
    Reshape parameter values from a numpy array to an array of shape (num_agents, *shape).

    Parameters
    ----------
    param_array : np.ndarray
        A NumPy array where each element is either a scalar or an array-like object.
    shape : tuple of int
        The desired shape for a single parameter value.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (num_agents, *shape).
    """

    num_agents = param_array.shape[0]

    if np.isscalar(param_array[0]):
        if len(shape) == 2 and shape[0] == shape[1]:
            d = shape[0]
            result = np.zeros((num_agents, d, d), dtype=np.result_type(param_array))
            diag_indices = np.arange(d)
            result[:, diag_indices, diag_indices] = param_array.reshape(num_agents, 1)
        else:
            reshaped = param_array.reshape((num_agents,) + (1,) * len(shape))
            result = np.broadcast_to(reshaped, (num_agents,) + shape)
    else:
        first_elem = np.array(param_array[0])
        if first_elem.shape != shape:
            raise ValueError(f"Expected shape {shape} but got {first_elem.shape} for the first element.")
        result = np.stack(param_array)

    return result



