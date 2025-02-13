import yaml
import progressbar

from swarmsim.Simulators import Simulator


class GymSimulator(Simulator):

    def __init__(self, populations, interactions, environment, integrator, logger, renderer, render_mode=None,
                 config_path=None) -> None:

        """        
        Initializes the Simulator class with configuration parameters from a YAML file.

        Args:
            config_path (str): The path to the YAML configuration file.
        """

        # Load config params from YAML file
        super().__init__(populations=populations,
                         interactions=interactions,
                         environment=environment,
                         controllers=None,
                         integrator=integrator,
                         logger=logger,
                         renderer=renderer,
                         config_path=config_path)

        self.render_mode = render_mode     # IMPLEMENT RENDER MODE

    def reset(self):

        # RESET INITIAL CONDITIONS OF THE POPULATIONS AND ENVIRONMENT AND LOGGER
        for population in self.populations:
            population.reset_state()
        self.logger.reset()

    def step(self, action):

        self.populations[1].u = action  # the first population is assumed to be controlled

        # Compute the interactions between the agents
        for interact in self.interactions:
            interact.pop1.f += interact.get_interaction()

        # Update the state of the agents
        self.integrator.step(self.populations)

        # Reset interaction forces
        for population in self.populations:
            population.f = 0

        # Update the environment
        self.environment.update()

    def render(self):
        if self.render_mode == "rgb_array":
            return self.renderer.render()
        else:
            self.renderer.render()

    def close(self):
        self.renderer.close()
