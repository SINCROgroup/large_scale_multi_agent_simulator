import yaml
import progressbar


class Simulator:

    def __init__(self, agents, environment, controller, integrator, logger, render, config_path) -> None:
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
        self.agents = agents
        self.environment = environment
        self.controller = controller
        self.logger = logger
        self.render = render
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

        u = 0
        f = 0
        self.logger.reset()
        self.logger.log(self.agents.x, u, f, self.environment)
        self.render.render(self.agents, self.environment)
        for t in range(num_steps):

            # print(f'step {t}')

            #AGENTS ha un campo u e un campo f . U rappresenta l'azione di controllo e f le interazioni con altri agenti e con l env
            for c in controllers:
                c.pop.u = c.get_action()   #COntroller ha un membro che è la popolazione su cui agisce
            
            #SOLUTION 2
            for i in int_list:
                i.pop1.f += i.get_interaction()  #Interaction ha 2 membri, pop1 subisce e pop2 da



            

            self.integrator.step(self.agents)
            # Update the environment

            # Execute every N steps
            self.logger.log(self.agents.x, u, f, self.environment)
            self.render.render(self.agents, self.environment)

            bar.update(t)
        self.logger.close()



