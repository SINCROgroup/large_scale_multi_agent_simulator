import numpy as np
from swarmsim.Populations import Population


class FixedPopulation(Population):
    """
    Stationary population with agents at fixed spatial locations.

    This population model implements immobile agents that remain at their initial
    positions throughout the simulation. The population is useful for modeling
    static obstacles, landmarks, reference points, or stationary infrastructure
    elements that other populations can interact with.

    The system dynamics are trivial:

    .. math::

        \\frac{d\\mathbf{x}}{dt} = \\mathbf{u}

    .. math::

        \\text{diffusion} = \\mathbf{0}

    where:
    - :math:`\\mathbf{x}` remains constant (typically :math:`\\mathbf{u} = \\mathbf{0}`)
    - No stochastic component affects the population

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file containing population parameters.

    Attributes
    ----------
    x : np.ndarray, shape (N, state_dim)
        Fixed positions of agents, set during initialization and typically unchanged.
    N : int
        Number of stationary agents in the population.
    f : np.ndarray, shape (N, state_dim)
        External forces (typically ignored for fixed populations).
    u : np.ndarray, shape (N, state_dim)
        Control input (typically zero for truly fixed populations).
    state_dim : int
        Spatial dimension of the agent positions.

    Config Requirements
    -------------------
    The YAML configuration file must contain the following parameters under the population's section:

    x0_mode : str
        Initial condition mode. Options: ``"From_File"``, ``"Random"``

    If ``x0_mode="From_File"``:
        x0_file_path : str
            Path to CSV file containing exact positions for stationary agents.
            This is the preferred method for precisely placed obstacles or landmarks.

    If ``x0_mode="Random"``:
        N : int
            Number of stationary agents to place randomly.
        state_dim : int
            Spatial dimension (2 for 2D, 3 for 3D).

    id : str, optional
        Population identifier. Defaults to class name.

    Notes
    -----
    **Use Cases:**

    Fixed populations are commonly used for:

    - **Static Obstacles**: Walls, barriers, immobile hazards
    - **Landmarks**: Navigation reference points, beacons
    - **Infrastructure**: Buildings, stations, charging points
    - **Environmental Features**: Trees, rocks, geographic features
    - **Sensors**: Fixed monitoring stations, cameras
    - **Reference Points**: Target locations, waypoints

    **Interaction Considerations:**

    While fixed populations don't move, they can:

    - Exert forces on other populations (through interactions)
    - Serve as sources of attraction or repulsion
    - Act as collision/avoidance targets
    - Provide spatial references for navigation

    """

    def get_drift(self) -> np.array :
        '''
        No movement on average.

        Returns:
        ------
            drift: numpy array (num_agents x dim_state)         Dirft of the population

        '''
        return self.u

    def get_diffusion(self) -> np.array :
        '''
        No stochasticity.

        Returns:
        ------
            diffusion: numpy array (num_agents x dim_state)     Diffusion of the population

        '''
        
        return np.zeros(self.x.shape)
