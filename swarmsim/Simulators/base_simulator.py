import yaml
import progressbar


class Simulator:

    def __init__(self, populations, interactions, environment, controllers, integrator, logger, renderer,
                 config_path) -> None:

        """        
        Initializes the Simulator class with configuration parameters from a YAML file.

        Args:
            config_path (str): The path to the YAML configuration file.
        """

        # Load config params from YAML file
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

        simulator_config = config.get('simulator', {})
        self.dt = simulator_config.get('dt', 0.01)
        self.T = simulator_config.get('T', 10)

        # get parameters from initialization
        self.populations = populations
        self.environment = environment
        self.interactions = interactions
        self.controllers = controllers
        self.logger = logger
        self.renderer = renderer
        self.integrator = integrator

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

            #Render the Scene if a rederer is defined
            if not(self.renderer==None):
                self.renderer.render()

            # Implement the control actions
            if self.controllers is not None:
                for c in self.controllers:
                    if (t % (round(c.dt / self.dt))) == 0:
                        c.population.u = c.get_action()  # Controller ha un membro che è la popolazione su cui agisce

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

            # Execute every N steps
            bar.update(t)

            if done:  # Truncate early the simulation
                print('\nSimulation truncated')
                break

        self.logger.close()
        if not(self.renderer==None):
            self.renderer.render()
