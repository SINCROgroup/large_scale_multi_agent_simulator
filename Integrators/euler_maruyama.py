import numpy as np
from Integrators.base_integrator import Integrator


class EulerMaruyamaIntegrator(Integrator):
    """
    Euler-Maruyama Integrator for stochastic differential equations (SDEs).
    This class implements the `step` method to integrate the state of an agent using the Euler-Maruyama method.
    """

    def __init__(self, config_path):
        """
        Initializes the Euler-Maruyama Integrator with configuration parameters from a YAML file.

        Args:
            config_path (str): The path to the YAML configuration file.
        """
        super().__init__(config_path)

    def step(self, agent, control_input, env_force):
        """
        Performs a single integration step using the Euler-Maruyama method.

        Args:
            agent: The agent for which the integration step is being performed. The agent must have `drift` and `diffusion` methods.
            control_input: The control input applied to the agent.
            env_force: The environmental force acting on the agent.
        """
        # Compute the drift and diffusion terms
        drift = agent.get_drift(control_input, env_force)
        diffusion = agent.get_diffusion(control_input, env_force)

        # Generate random noise
        noise = np.random.normal(0, 1, size=agent.x.shape)

        # Update agent state using Euler-Maruyama method
        agent.x = agent.x + drift * self.dt + diffusion * np.sqrt(self.dt) * noise
