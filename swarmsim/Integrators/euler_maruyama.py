import numpy as np
from swarmsim.Integrators import Integrator


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

    def step(self, populations):
        """
        Performs a single integration step using the Euler-Maruyama method.

        Args:
            populations: List of populations for which the integration step is being performed.
                         Each population must have `drift` and `diffusion` methods.
        """
        # Compute the drift and diffusion terms

        for population in populations:

            drift = population.get_drift()
            diffusion = population.get_diffusion()

            # Generate random noise
            noise = np.random.normal(0, 1, size=population.x.shape)

            # Update agent state using Euler-Maruyama method
            population.x = population.x + drift * self.dt + diffusion * np.sqrt(self.dt) * noise
