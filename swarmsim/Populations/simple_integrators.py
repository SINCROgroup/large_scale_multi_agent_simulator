import numpy as np
from swarmsim.Populations import Population


class SimpleIntegrators(Population):
    """
    First-order integrator dynamics with velocity constraints.

    This population model implements simple first-order integrator dynamics where
    the rate of change of agent positions is directly controlled by the sum of
    control inputs and external forces, subject to velocity magnitude constraints.
    This model is suitable for kinematic systems or overdamped dynamics.

    The system dynamics are governed by:

    .. math::

        \\frac{d\\mathbf{x}}{dt} = \\text{clip}(\\mathbf{u} + \\mathbf{f}, -v_{max}, v_{max})

    where:
    - :math:`\\mathbf{x}` is the agent position/state vector
    - :math:`\\mathbf{u}` is the control input (desired velocity)
    - :math:`\\mathbf{f}` is the external force/influence
    - :math:`v_{max}` is the maximum allowed velocity magnitude per component

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file containing population parameters.
    name : str, optional
        Name identifier for the population. Defaults to class name if None.

    Attributes
    ----------
    x : np.ndarray, shape (N, state_dim)
        Current state/position of agents.
    v_max : float
        Maximum allowed velocity magnitude per state component.
        Default: ``float('inf')`` (no velocity limit).
    N : int
        Number of agents in the population, inherited from Population.
    f : np.ndarray, shape (N, state_dim)
        External forces/influences applied to agents, inherited from Population.
    u : np.ndarray, shape (N, state_dim)
        Control inputs (desired velocities), inherited from Population.
    state_dim : int
        Dimension of the state space, inherited from Population.

    Config Requirements
    -------------------
    The YAML configuration file must contain the following parameters under the population's section:

    initial_conditions : dict
        Initial conditions for the population, including positions and velocities.

    parameters : dict
        Parameter configuration for damping and diffusion:

    id : str, optional
        Population identifier. Defaults to class name.

    N : int
        Number of agents in the population.

    state_dim : int
        Dimension of the state space, inherited from Population.

    v_max : float, optional
        Maximum velocity magnitude per component. Default: ``inf`` (no limit).


    Notes
    -----
    **System Characteristics:**

    - **No Inertia**: Agents respond instantaneously to control inputs
    - **Velocity Saturation**: Motion is bounded by velocity constraints
    - **Deterministic**: No stochastic diffusion (overdamped regime)
    - **Direct Control**: Control input directly sets velocity (subject to limits)

    Examples
    --------
    **Basic Configuration:**

    .. code-block:: yaml

        SimpleIntegrators:
            x0_mode: "Random"
            N: 100
            state_dim: 2
            v_max: 2.0
            id: "mobile_robots"

    """

    def __init__(self, config_path: str, name: str = None) -> None:
        super().__init__(config_path, name)

        self.v_max = self.config.get('v_max', float('inf'))  # Action limit

    def get_drift(self):
        return np.clip(self.u + self.f, -self.v_max, self.v_max)

    def get_diffusion(self):
        return np.zeros((self.N, self.state_dim))
