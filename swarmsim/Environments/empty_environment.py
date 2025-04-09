import numpy as np
from swarmsim.Environments import Environment


class EmptyEnvironment(Environment):
    """
    An environment with no forces acting on the agents.

    This environment represents a simple **static** environment where agents
    are not influenced by external forces. The environment dimensions are loaded
    from a YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file containing environment parameters.

    Attributes
    ----------
    dimensions : tuple of (int, int)
        The dimensions of the environment in 2D (width, height), inherited from the `Environment` base class.

    Config requirements
    -------------------
    dimensions : tuple of (int, int), optional
        The dimensions of the environment in 2D. Default is ``(50, 50)``.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    KeyError
        If required environment parameters are missing in the configuration file.

    Examples
    --------
    Example YAML configuration:

    .. code-block:: yaml

        environment:
            dimensions: [50, 50]

    This will set the environment dimensions to ``(50, 50)``.
    """

    def __init__(self, config_path):
        """
        Initializes the EmptyEnvironment with the configuration parameters from a YAML file.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file.
        """
        super().__init__(config_path)

    def get_forces(self, agents):
        """
        Computes the forces exerted by the environment on the agents.

        Since this is an **empty environment**, no external forces are exerted on the agents.
        This method returns an array of zeros, representing zero force applied to each agent.

        Parameters
        ----------
        agents : list
            A list of agent objects for which the environmental forces are being computed.

        Returns
        -------
        np.ndarray
            An array of shape `(num_agents, 2)`, where each row is `[0, 0]` indicating no force.
        """
        return np.zeros((agents.x.shape[0], 2))  # No forces, so return zero vectors for each agent

    def get_info(self):
        """
        Retrieves environment-specific information for logging.

        Since this is an empty environment, it returns an empty dictionary.

        Returns
        -------
        dict
            An empty dictionary `{}`.
        """
        return {}

    def update(self):
        """
        Updates the environment state.

        Since this is a static environment with no forces or dynamic elements,
        this method does nothing.
        """
        pass
