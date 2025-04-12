from swarmsim.Utils import load_config
import progressbar
import numpy as np

from swarmsim.Environments import Environment
from swarmsim.Populations import Populations
from swarmsim.Interactions import Interaction
from swarmsim.Controllers import Controller
from swarmsim.Integrators import Integrator
from swarmsim.Loggers import Logger
from swarmsim.Renderers import Renderer


class Simulator:

    def __init__(self,
                 config_path: str,
                 populations: list[Populations],
                 environment: Environment,
                 integrator: Integrator,
                 interactions: list[Interaction] | None =None,
                 controllers: list[Controller] | None =None,
                 logger: Logger | None =None,
                 renderer: Renderer | None =None,
                 ) -> None:

        """        
        Initializes the Simulator class with configuration parameters from a YAML file.

        Args:
            config_path (str): The path to the YAML configuration file.
        """

        config: dict = load_config(config_path)

        simulator_config: dict = config.get('simulator', {})
        self.dt: float = integrator.dt
        self.T: float = simulator_config.get('T', 10)

        # get parameters from initialization
        self.populations: list[Populations] = populations
        self.environment: Environment = environment
        self.interactions: list[Interaction] | None = interactions
        self.controllers: list[Controller] | None = controllers
        self.logger: Logger | None = logger
        self.renderer: Renderer | None = renderer
        self.integrator: Integrator = integrator

    def simulate(self):

        num_steps = int(self.T / self.dt)  # Calculate the number of steps as an integer

        bar = progressbar.ProgressBar(
            max_value=num_steps,
            widgets=[
                'Processing: ',  # Custom description
                progressbar.Percentage(),
                ' ', progressbar.Bar(marker='=', left='[', right=']'),
                ' ', progressbar.ETA()
            ]
        )

        self.logger.reset()
        
        for t in range(num_steps):
            #print(t)
            done = self.logger.log()  # Log and get done flag

            # Render the Scene if a renderer is defined
            if self.renderer is not None:
                self.renderer.render()

            # Implement the control actions
            if self.controllers is not None:
                for c in self.controllers:
                    if (t % (round(c.dt / self.dt))) == 0:
                        c.population.u = c.get_action()

            # Compute the interactions between the agents
            if self.interactions is not None:
                for interact in self.interactions:
                    interact.target_population.f += interact.get_interaction()

            # Update the state of the agents
            self.integrator.step(self.populations)

            # Reset interaction forces
            for population in self.populations:
                population.f = np.zeros([population.N, population.input_dim])

            # Update the environment
            self.environment.update()

            # Execute every N steps
            bar.update(t)

            if done:  # Truncate early the simulation
                print('\nSimulation truncated')
                break

        self.logger.close()
        if self.renderer is not None:
            self.renderer.render()
