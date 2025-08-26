import numpy as np
from swarmsim.Populations import Population
from typing import Optional


class DampedDoubleIntegrators(Population):
    """
    Damped second-order integrator dynamics for agent populations.

    This population model implements damped double integrator dynamics, where each agent
    has both position and velocity states. The dynamics include damping, external forces,
    control inputs, and stochastic noise. This model is commonly used for modeling
    vehicles, robots, or particles with inertial behavior.

    The system dynamics are governed by:

    .. math::

        \\frac{d}{dt} \\begin{bmatrix} \\mathbf{p} \\\\ \\mathbf{v} \\end{bmatrix} = 
        \\begin{bmatrix} \\mathbf{v} \\\\ -\\gamma \\mathbf{v} + \\mathbf{u} + \\mathbf{f} \\end{bmatrix} + 
        \\mathbf{D} \\, d\\mathbf{W}

    where:
    - :math:`\\mathbf{p}` is the position vector
    - :math:`\\mathbf{v}` is the velocity vector  
    - :math:`\\gamma` is the damping coefficient
    - :math:`\\mathbf{u}` is the control input
    - :math:`\\mathbf{f}` is the external force
    - :math:`\\mathbf{D}` is the diffusion matrix
    - :math:`d\\mathbf{W}` is white noise

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file containing population parameters.
    name : str, optional
        Name identifier for the population. Defaults to class name if None.

    Attributes
    ----------
    x : np.ndarray, shape (N, state_dim)
        Current state of agents with positions and velocities concatenated.
        Format: ``[pos_x, pos_y, vel_x, vel_y]`` for 2D systems.
    damping : np.ndarray, shape (N,) or None
        Damping coefficient for each agent affecting velocity decay.
    D : np.ndarray, shape (N, state_dim) or None
        Diffusion coefficients for position and velocity noise.
        Typically zero for positions, non-zero for velocities.
    N : int
        Number of agents in the population, inherited from Population.
    f : np.ndarray, shape (N, state_dim)
        External forces applied to agents, inherited from Population.
    u : np.ndarray, shape (N, state_dim)
        Control inputs applied to agents, inherited from Population.
    input_dim : int
        Dimension of the spatial coordinates (typically 2 for 2D systems).
    params_shapes : dict
        Defines expected shapes for population parameters.

    Config Requirements
    -------------------
    The YAML configuration file must contain the following parameters under the population's section:

    initial_conditions : dict
        Initial conditions for the population, including positions and velocities.

    parameters : dict
        Parameter configuration for damping and diffusion:

    id : str, optional
        Population identifier. Defaults to class name.

    state_dim : int
        Dimension of the state space, inherited from Population.

    N : int
        Number of agents in the population.


    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    KeyError
        If required parameters are missing from the configuration.
    ValueError
        If parameter shapes are incompatible with the population size.

    Examples
    --------
    **Basic Configuration:**

    .. code-block:: yaml

        DampedDoubleIntegrators:
            x0_mode: "Random"
            N: 50
            state_dim: 2
            id: "vehicles"
            parameters:
                params_mode: "Fixed"
                params_names: ["damping", "D"]
                params_values:
                    damping: 0.1
                    D: 0.05
    """


    def __init__(self, config_path: str, name: str = None) -> None:
        super().__init__(config_path, name)

        self.damping: Optional[np.ndarray] = None
        self.D: Optional[np.ndarray] = None

        self.params_shapes = {
            'damping': (),
            'D': ()
        }

    def reset(self) -> None:
        """
        Resets the state of the population to its initial conditions.

        This method reinitializes the agent states, external forces, and control inputs.
        """
        super().reset()

        self.damping = self.params['damping']
        self.D = self.params['D'][:,np.newaxis] * np.array([0, 0, 1, 1])

    def get_drift(self):
        """
        Compute the deterministic drift term for damped double integrator dynamics.

        This method implements the drift component of the stochastic differential equation
        governing damped double integrator motion. The drift includes velocity propagation
        to position and damped acceleration dynamics with control and external forces.

        Returns
        -------
        np.ndarray, shape (N, state_dim)
            Drift vector for all agents, containing velocity and acceleration terms.
            Format: ``[velocity_x, velocity_y, acceleration_x, acceleration_y]`` for 2D.

        Notes
        -----
        **Drift Structure:**

        The drift term consists of two components:

        1. **Position Drift**: Current velocity becomes position rate of change
        2. **Velocity Drift**: Damped acceleration with control and external forces

        **Mathematical Form:**

        .. math::

            \\frac{d\\mathbf{p}}{dt} = \\mathbf{v}

            \\frac{d\\mathbf{v}}{dt} = -\\gamma \\mathbf{v} + \\mathbf{u} + \\mathbf{f}

        where:
        - :math:`\\mathbf{p}` is position
        - :math:`\\mathbf{v}` is velocity
        - :math:`\\gamma` is damping coefficient
        - :math:`\\mathbf{u}` is control input  
        - :math:`\\mathbf{f}` is external force

        **Damping Effects:**

        The damping term :math:`-\\gamma \\mathbf{v}` provides:
        - Velocity-dependent resistance (like air resistance)
        - Exponential decay toward equilibrium
        - Stabilization of the system dynamics

        **Force Integration:**

        The method combines control and external forces:
        - **Control Forces**: Deliberate actuator inputs
        - **External Forces**: Environmental influences and inter-agent interactions

        """

        velocity = self.x[:, self.input_dim:]  # current velocities
        acceleration = -self.damping[:, np.newaxis] * velocity + self.u + self.f

        return np.hstack((velocity, acceleration))

    def get_diffusion(self):
        """
        Compute the diffusion term for stochastic double integrator dynamics.

        This method returns the diffusion matrix that scales the white noise in the
        stochastic differential equation. For double integrators, noise typically
        affects velocities rather than positions directly, modeling random accelerations
        from unmodeled forces or actuator noise.

        Returns
        -------
        np.ndarray, shape (N, 2*state_dim)
            Diffusion coefficients for all agents and state components.
            Typically structured as ``[0, 0, D_vel, D_vel]`` for 2D systems.
        
        """
        return self.D


