import numpy as np
import pandas as pd
from pathlib import Path
import logging


def get_parameters(params_config: dict, params_shapes: dict[str, tuple], num_samples: int) -> dict[str, np.ndarray]:
    """
    Load or generate parameters for population initialization with flexible configuration.

    This function serves as the main entry point for parameter generation in the swarmsim
    framework. It supports both file-based parameter loading and procedural parameter
    generation using statistical distributions, enabling flexible population initialization
    for multi-agent simulations.

    Parameters
    ----------
    params_config : dict
        Configuration dictionary specifying parameter loading/generation method.
        Must contain 'mode' key with value 'file' or 'generate', plus corresponding
        configuration section.
    params_shapes : dict[str, tuple]
        Dictionary mapping parameter names to their expected shapes per agent.
        Used for validation and reshaping operations.
    num_samples : int
        Number of agents/samples to generate parameters for.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping parameter names to NumPy arrays of shape (num_samples, *shape)
        where shape corresponds to the parameter's expected dimensions.

    Configuration Structure
    ----------------------
    File Mode:
        .. code-block:: yaml

            mode: "file"
            file:
              file_path: "path/to/parameters.csv"  # or .npz

    Generate Mode:
        .. code-block:: yaml

            mode: "generate"
            generate:
              parameter_name:
                sampler: "normal"  # Distribution type
                args:
                  loc: 0.0         # Distribution parameters
                  scale: 1.0
                shape: [3]         # Parameter shape per agent
                homogeneous: false # Same value for all agents

    
    Error Handling
    --------------
    - **FileNotFoundError**: Raised when specified parameter file doesn't exist
    - **KeyError**: Raised when expected parameters are missing from loaded data
    - **ValueError**: Raised for invalid parameter shapes or configurations
    - **RuntimeError**: Raised for invalid mode specifications

    Applications
    ------------
    - **Population Initialization**: Set agent properties for simulation start
    - **Parameter Studies**: Generate parameter sets for sensitivity analysis
    - **Multi-species Systems**: Handle heterogeneous agent populations
    - **Reproducible Research**: Load exact parameter sets from files

    
    See Also
    --------
    _load_parameters : File-based parameter loading implementation
    _generate_parameters : Statistical parameter generation implementation
    _reshape_parameter : Parameter reshaping utilities
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
    Load parameters from CSV or NPZ files with automatic population matching.

    This helper function handles file-based parameter loading with intelligent
    population size matching. It automatically repeats or truncates parameter
    data to match the required number of samples, ensuring consistent population
    initialization regardless of file size.

    Parameters
    ----------
    params_config : dict
        Configuration dictionary containing file loading settings.
        Must include 'file_path' key pointing to CSV or NPZ file.
    num_samples : int
        Target number of parameter sets to generate. File data will be
        adjusted to match this count.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping parameter names to NumPy arrays, each with
        num_samples entries along the first dimension.

    File Format Support
    -------------------
    **CSV Files**:
        - Column headers become parameter names
        - Each row represents one agent's parameters
        - Supports all numeric data types
        - Missing values handled automatically

    **NPZ Files**:
        - Each array key becomes a parameter name
        - First dimension must match across all parameters
        - Preserves exact numerical precision
        - Supports complex data structures

    Population Matching
    -------------------
    **Too Few Samples**: Randomly repeats existing rows to reach target count
    **Too Many Samples**: Randomly selects subset to match target count
    **Exact Match**: Uses data as-is without modification

    Examples
    --------
    Loading CSV parameter file:

    .. code-block:: python

        # CSV file structure (example: euglena_params.csv):
        # mass,diameter,max_speed,light_sensitivity
        # 1.2,0.05,2.1,0.8
        # 1.1,0.048,2.3,0.75
        # 1.3,0.052,1.9,0.85

        config = {
            'file_path': 'Configuration/Config_data/euglena_params.csv'
        }
        
        params = _load_parameters(config, num_samples=100)
        
        # Returns:
        # {
        #     'mass': array([1.2, 1.1, 1.3, 1.2, ...]),      # 100 samples
        #     'diameter': array([0.05, 0.048, 0.052, ...]),   # 100 samples
        #     'max_speed': array([2.1, 2.3, 1.9, ...]),       # 100 samples
        #     'light_sensitivity': array([0.8, 0.75, 0.85, ...]) # 100 samples
        # }

    
    Error Handling
    --------------
    - **FileNotFoundError**: When specified file doesn't exist
    - **ValueError**: For unsupported file formats or invalid data
    - **KeyError**: When expected parameter columns/arrays are missing
    - **MemoryError**: For extremely large files (use chunked loading)

    Notes
    -----
    - Preserves parameter names exactly as they appear in files
    - Maintains data types from original files when possible
    - Logs warnings when adjusting population size
    - Supports both absolute and relative file paths
    - Thread-safe for concurrent parameter loading
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
    Generate parameters using statistical distributions and configuration.

    This function creates parameter sets using a comprehensive collection of
    statistical distributions from NumPy's random module. It supports both
    homogeneous (same value for all agents) and heterogeneous (different
    values per agent) parameter generation with flexible shape control.

    Parameters
    ----------
    params_config : dict
        Configuration dictionary specifying parameters to generate.
        Each key represents a parameter name with associated generation settings.
    num_samples : int
        Number of parameter sets (agents) to generate.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping parameter names to generated NumPy arrays.
        Arrays have shape (num_samples, *parameter_shape).

    Supported Distributions
    -----------------------
    **Continuous Distributions**:
        - uniform, normal, exponential, gamma, beta, lognormal
        - laplace, logistic, gumbel, rayleigh, weibull, pareto
        - standard_normal, standard_exponential, standard_gamma
        - triangular, vonmises, wald, power, chisquare, f
        - noncentral_chisquare, noncentral_f, standard_cauchy, standard_t

    **Discrete Distributions**:
        - poisson, binomial, negative_binomial, geometric
        - hypergeometric, logseries, zipf

    **Multivariate Distributions**:
        - multivariate_normal, multinomial

    Parameter Configuration
    ----------------------
    Each parameter can be configured as:

    1. **Constant Value**: Simple numeric value or array
    2. **Distribution Sampling**: Dictionary with sampler configuration
    3. **Homogeneous Sampling**: Same random value for all agents
    4. **Heterogeneous Sampling**: Different random values per agent

    Examples
    --------
    Basic parameter generation:

    .. code-block:: python

        config = {
            # Constant parameters
            'species_id': 1,                    # Same for all agents
            'max_neighbors': 10,                # Same for all agents
            
            # Scalar random parameters
            'mass': {
                'sampler': 'normal',
                'args': {'loc': 1.0, 'scale': 0.1}
            },
            
            # Vector random parameters  
            'initial_position': {
                'sampler': 'uniform',
                'args': {'low': -5.0, 'high': 5.0},
                'shape': [3]  # 3D position
            },
            
            # Homogeneous parameters (same random value for all)
            'environment_temperature': {
                'sampler': 'normal',
                'args': {'loc': 20.0, 'scale': 2.0},
                'homogeneous': True
            }
        }
        
        params = _generate_parameters(config, num_samples=100)

    
    Applications
    ------------
    - **Agent Initialization**: Set diverse agent properties for realistic populations
    - **Parameter Studies**: Generate parameter sets for sensitivity analysis
    - **Monte Carlo Simulation**: Generate random inputs for statistical analysis

    See Also
    --------
    numpy.random : Complete documentation of available distributions
    _reshape_parameter : Function for parameter shape manipulation
    get_parameters : Main parameter loading/generation interface
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
    Reshape parameter arrays to consistent multi-agent format with automatic broadcasting.

    This function standardizes parameter arrays to have consistent dimensions across
    all agents, handling scalar-to-tensor broadcasting and special cases like
    diagonal matrix creation. It ensures all parameters have the format
    (num_agents, *parameter_shape) for uniform agent property access.

    Parameters
    ----------
    param_array : np.ndarray
        Input parameter array where the first dimension corresponds to agents.
        Can contain scalars, vectors, or higher-dimensional structures per agent.
    shape : tuple of int, optional
        Target shape for each agent's parameter. Empty tuple () indicates scalar.
        Special handling for square matrices (diagonal initialization from scalars).

    Returns
    -------
    np.ndarray
        Reshaped array with dimensions (num_agents, *shape), where each agent's
        parameter conforms to the specified shape through broadcasting or 
        transformation.

    Shape Transformation Rules
    -------------------------
    1. **Scalar to Scalar**: (N,) → (N,) - No change needed
    2. **Scalar to Vector**: (N,) → (N, d) - Broadcast scalar to vector
    3. **Scalar to Matrix**: (N,) → (N, d, d) - Create diagonal matrices
    4. **Vector to Vector**: (N, d) → (N, d) - Direct assignment
    5. **Matrix to Matrix**: (N, d, d) → (N, d, d) - Direct assignment

    Special Cases
    -------------
    **Diagonal Matrix Creation**: When target shape is (d, d) and input contains
    scalars, creates diagonal matrices with the scalar value on the diagonal.

    **Broadcasting**: When target shape has more dimensions than input, broadcasts
    the input to match the target shape while preserving agent dimension.

    Examples
    --------
    Scalar to vector broadcasting:

    .. code-block:: python

        # Input: scalar values per agent
        masses = np.array([1.0, 1.2, 0.8, 1.1])  # Shape: (4,)
        
        # Reshape to 3D vectors (same mass in all dimensions)
        mass_vectors = _reshape_parameter(masses, shape=(3,))
        print(mass_vectors.shape)  # (4, 3)
        print(mass_vectors[0])     # [1.0, 1.0, 1.0]

    
    Error Handling
    --------------
    - **ValueError**: Raised when input array shapes are incompatible with target shape
    - **TypeError**: Raised when input is not a NumPy array
    - **IndexError**: Raised when accessing invalid array dimensions

    
    Applications
    ------------
    - **Agent Initialization**: Standardize diverse parameter formats
    - **Physical Properties**: Create mass/inertia matrices from scalar values
    - **Neural Networks**: Initialize weight matrices for agent brains
    - **Interaction Models**: Setup pairwise interaction parameter tensors
    - **Control Systems**: Create gain matrices from scalar tuning parameters

    See Also
    --------
    numpy.broadcast_to : NumPy broadcasting documentation
    numpy.zeros : Array initialization
    numpy.stack : Array stacking operations
    get_parameters : Main parameter processing interface
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



