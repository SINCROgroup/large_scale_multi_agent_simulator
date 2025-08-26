import numpy as np
import yaml
from swarmsim.Populations import Population
from typing import Optional


class LightSensitive_PTW(Population):
    """
    Light-sensitive Persistent Turning Walker population model.

    This population implements a sophisticated bio-inspired locomotion model based on 
    persistent turning walkers that respond to external light stimuli. The model captures
    the complex dynamics of motile organisms (like Euglena) that exhibit persistent
    motion with turning behavior modulated by environmental light conditions.

    The system state consists of position, velocity magnitude, orientation, and angular
    velocity, governed by the following dynamics:

    .. math::

        \\frac{dx}{dt} = v \\cos(\\theta)

        \\frac{dy}{dt} = v \\sin(\\theta)

        \\frac{dv}{dt} = \\theta_s(\\mu_s - v) + \\alpha_s u + \\beta_s \\dot{u}^+ + \\gamma_s \\dot{u}^- + \\sigma_s \\, dW_v

        \\frac{d\\theta}{dt} = \\omega

        \\frac{d\\omega}{dt} = \\theta_w(\\mu_w - \\omega) + \\text{sign}(\\omega)(\\alpha_w u + \\beta_w \\dot{u}^+ + \\gamma_w \\dot{u}^-) + \\sigma_w \\, dW_\\omega

    where:
    - :math:`(x, y)` is the agent position
    - :math:`v` is the speed magnitude  
    - :math:`\\theta` is the orientation angle
    - :math:`\\omega` is the angular velocity
    - :math:`u` is the light stimulus intensity
    - :math:`\\dot{u}^+, \\dot{u}^-` are positive and negative stimulus derivatives
    - :math:`\\theta_s, \\mu_s, \\alpha_s, \\beta_s, \\gamma_s, \\sigma_s` are speed parameters
    - :math:`\\theta_w, \\mu_w, \\alpha_w, \\beta_w, \\gamma_w, \\sigma_w` are turning parameters

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file containing population parameters.
    name : str, optional
        Name identifier for the population. Defaults to class name if None.

    Attributes
    ----------
    x : np.ndarray, shape (N, 5)
        State of agents: ``[x_pos, y_pos, velocity, theta, omega]``.
    u_old : np.ndarray, shape (N, input_dim) or None
        Previous timestep control input for computing derivatives.
    theta_s, mu_s, alpha_s, beta_s, gamma_s, sigma_s : np.ndarray or None
        Speed dynamics parameters for each agent.
    theta_w, mu_w, alpha_w, beta_w, gamma_w, sigma_w : np.ndarray or None
        Turning dynamics parameters for each agent.
    dt : float
        Integration timestep for computing input derivatives.
    params_shapes : dict
        Defines expected shapes for all model parameters.

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
    
    dt : float
        Integration timestep for derivative calculations.

    Notes
    -----
    **Biological Inspiration:**

    This model is inspired by the locomotion of *Euglena gracilis* and similar
    photo-responsive microorganisms that exhibit:

    - **Persistent Motion**: Sustained movement with characteristic speeds
    - **Phototaxis**: Movement in response to light gradients
    - **Turning Behavior**: Modulation of orientation based on stimuli
    - **Adaptive Responses**: Different reactions to increasing vs decreasing light

    **Parameter Interpretation:**

    **Speed Parameters:**
    - ``theta_s``: Speed relaxation rate (return to preferred speed)
    - ``mu_s``: Preferred/natural swimming speed
    - ``alpha_s``: Direct light response on speed
    - ``beta_s``: Response to increasing light intensity
    - ``gamma_s``: Response to decreasing light intensity
    - ``sigma_s``: Speed noise amplitude

    **Turning Parameters:**
    - ``theta_w``: Angular velocity relaxation rate
    - ``mu_w``: Preferred angular velocity (bias for turning)
    - ``alpha_w``: Direct light response on turning
    - ``beta_w``: Turning response to increasing light
    - ``gamma_w``: Turning response to decreasing light
    - ``sigma_w``: Angular noise amplitude

    **State Variables:**

    The 5D state vector represents:
    1. **x, y**: Spatial position coordinates
    2. **v**: Instantaneous speed magnitude (always ≥ 0)
    3. **θ**: Orientation angle (wrapped to [0, 2π])
    4. **ω**: Angular velocity (rate of orientation change)

    **Light Response Mechanism:**

    The model distinguishes between:
    - **Current stimulus**: Direct response to light intensity
    - **Positive changes**: Response to increasing light (photo-attraction)
    - **Negative changes**: Response to decreasing light (photo-avoidance)

    Examples
    --------
    **Basic Euglena-like Configuration:**

    .. code-block:: yaml

        LightSensitive_PTW:
            x0_mode: "Random"
            N: 100
            state_dim: 5
            dt: 0.01
            id: "euglena"
            parameters:
                params_mode: "Fixed"
                params_names: ["theta_s", "mu_s", "alpha_s", "beta_s", "gamma_s", "sigma_s",
                              "theta_w", "mu_w", "alpha_w", "beta_w", "gamma_w", "sigma_w"]
                params_values:
                    theta_s: 1.0
                    mu_s: 2.0
                    alpha_s: 0.5
                    beta_s: 1.0
                    gamma_s: -0.5
                    sigma_s: 0.1
                    theta_w: 2.0
                    mu_w: 0.0
                    alpha_w: 0.2
                    beta_w: 0.8
                    gamma_w: -0.3
                    sigma_w: 0.2
    """

    def __init__(self, config_path: str, name: str = None) -> None:

        super().__init__(config_path, name)

        self.u_old: Optional[np.ndarray] = None

        self.theta_s: Optional[np.ndarray] = None
        self.mu_s: Optional[np.ndarray] = None
        self.alpha_s: Optional[np.ndarray] = None
        self.beta_s: Optional[np.ndarray] = None
        self.gamma_s: Optional[np.ndarray] = None
        self.sigma_s: Optional[np.ndarray] = None

        self.theta_w: Optional[np.ndarray] = None
        self.mu_w: Optional[np.ndarray] = None
        self.alpha_w: Optional[np.ndarray] = None
        self.beta_w: Optional[np.ndarray] = None
        self.gamma_w: Optional[np.ndarray] = None
        self.sigma_w: Optional[np.ndarray] = None

        self.params_shapes = {
            'theta_s': (),
            'mu_s': (),
            'alpha_s': (),
            'beta_s': (),
            'gamma_s': (),
            'sigma_s': (),

            'theta_w': (),
            'mu_w': (),
            'alpha_w': (),
            'beta_w': (),
            'gamma_w': (),
            'sigma_w': ()
        }


        self.dt = self.config["dt"]

    def reset(self) -> None:
        """
        Resets the state of the population to its initial conditions.

        This method reinitializes the agent states, external forces, and control inputs.
        """
        super().reset()

        self.theta_s = self.params['theta_s']
        self.mu_s = self.params['mu_s']
        self.alpha_s = self.params['alpha_s']
        self.beta_s = self.params['beta_s']
        self.gamma_s = self.params['gamma_s']
        self.sigma_s = self.params['sigma_s']

        self.theta_w = self.params['theta_w']
        self.mu_w = self.params['mu_w']
        self.alpha_w = self.params['alpha_w']
        self.beta_w = self.params['beta_w']
        self.gamma_w = self.params['gamma_w']
        self.sigma_w = self.params['sigma_w']

        self.u_old = np.zeros([self.N, self.input_dim])  # Initialization of the last control input applied

    def get_drift(self) -> np.array:
        """
        Compute the deterministic drift for persistent turning walker dynamics.

        This method implements the complete drift dynamics for light-sensitive persistent
        turning walkers, including position propagation, speed regulation, orientation
        change, and angular velocity dynamics with light stimulus responses.

        Returns
        -------
        np.ndarray, shape (N, 5)
            Drift vector for all agents containing:
            ``[dx/dt, dy/dt, dv/dt, dθ/dt, dω/dt]``

        Notes
        -----
        **State Variable Processing:**

        The method extracts and processes state variables:

        - **Speed Constraint**: Ensures velocity magnitude is non-negative
        - **Angle Wrapping**: Keeps orientation in [0, 2π] range
        - **Derivative Computation**: Calculates input rate of change

        **Dynamics Components:**

        1. **Position Dynamics**: 
           :math:`\\frac{dx}{dt} = v \\cos(\\theta)`, :math:`\\frac{dy}{dt} = v \\sin(\\theta)`

        2. **Speed Dynamics**:
           :math:`\\frac{dv}{dt} = \\theta_s(\\mu_s - v) + \\alpha_s u + \\beta_s \\dot{u}^+ + \\gamma_s \\dot{u}^-`

        3. **Orientation Dynamics**:
           :math:`\\frac{d\\theta}{dt} = \\omega`

        4. **Angular Velocity Dynamics**:
           :math:`\\frac{d\\omega}{dt} = \\theta_w(\\mu_w - \\omega) + \\text{sign}(\\omega)[\\alpha_w u + \\beta_w \\dot{u}^+ + \\gamma_w \\dot{u}^-]`

        **Light Response Processing:**

        The method computes stimulus derivatives and separates positive/negative changes:

        - :math:`\\dot{u} = \\frac{u(t) - u(t-\\Delta t)}{\\Delta t}`
        - :math:`\\dot{u}^+ = \\max(\\dot{u}, 0)` (increasing light)
        - :math:`\\dot{u}^- = \\min(\\dot{u}, 0)` (decreasing light)

        **Sign-Dependent Turning:**

        Angular velocity dynamics include sign-dependent responses to maintain
        biological realism in turning behavior asymmetries.
        """

        # State Variables: x, y, v, theta, omega(w)

        
        #self.x[:,2] = np.clip(self.x[:,2], 0,None)  # Ensure speed is non-negative
        v = self.x[:,2]
        self.x[:,3] = np.mod(self.x[:,3], 2 * np.pi)  # Ensure theta is in [0, 2*pi]
        theta = self.x[:,3]
        w = self.x[:,4]

        du = (self.u - self.u_old)/self.dt
        du_pos = np.max(np.hstack((du,np.zeros(du.shape))),axis=1)
        du_neg = np.min(np.hstack((du,np.zeros(du.shape))),axis=1)
        self.u_old = self.u

        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dv = self.theta_s * (self.mu_s - v) + self.alpha_s * self.u[:,0] + self.beta_s * du_pos + self.gamma_s * du_neg
        dth = w
        dw = self.theta_w * (self.mu_w - w) + np.sign(w) * (self.alpha_w * self.u[:,0] + self.beta_w * du_pos + self.gamma_w * du_neg)


        drift = np.hstack((dx[:,np.newaxis],dy[:,np.newaxis],dv[:,np.newaxis],dth[:,np.newaxis],dw[:,np.newaxis]))
        return drift

    def get_diffusion(self) -> np.array:
        """
        Compute the diffusion matrix for stochastic persistent turning walker dynamics.

        This method returns the diffusion coefficients that scale white noise in the
        stochastic differential equation. For persistent turning walkers, noise affects
        the speed and angular velocity dynamics while position and orientation change
        deterministically through their respective rates.

        Returns
        -------
        np.ndarray, shape (N, 5)
            Diffusion coefficients for all state variables:
            ``[0, 0, σ_s, 0, σ_w]`` where only speed and angular velocity have noise.

        Notes
        -----
        **Noise Structure:**

        The diffusion reflects the biological noise sources:

        - **Position (x, y)**: Zero noise - positions change deterministically via velocity
        - **Speed (v)**: ``σ_s`` noise - models metabolic fluctuations, swimming variability
        - **Orientation (θ)**: Zero noise - angle changes deterministically via angular velocity  
        - **Angular Velocity (ω)**: ``σ_w`` noise - models turning decision variability

        """

        dx = np.zeros([self.N,1])
        dy = np.zeros([self.N,1])
        dv = self.sigma_s
        dth = np.zeros([self.N,1])
        dw = self.sigma_w

        diffusion = np.hstack((dx,dy,dv[:,np.newaxis],dth,dw[:,np.newaxis]))

        return diffusion


