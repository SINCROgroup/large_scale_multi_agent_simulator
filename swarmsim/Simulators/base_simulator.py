from swarmsim.Utils import load_config
import progressbar
import numpy as np

from swarmsim.Environments import Environment
from swarmsim.Populations import Population
from swarmsim.Interactions import Interaction
from swarmsim.Controllers import Controller
from swarmsim.Integrators import Integrator
from swarmsim.Loggers import Logger
from swarmsim.Renderers import Renderer


class Simulator:
    """
    Main simulation engine that orchestrates multi-agent system simulations.

    The Simulator class coordinates all components of a multi-agent system including populations,
    environment, interactions, controllers, integrators, loggers, and renderers. It manages the
    simulation loop and ensures proper execution order of all simulation steps.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file containing simulation parameters.
    populations : list of Population
        List of agent populations to simulate.
    environment : Environment
        The environment in which agents operate.
    integrator : Integrator
        Numerical integration scheme for updating agent states.
    interactions : list of Interaction, optional
        List of interaction models between agents. Default is None.
    controllers : list of Controller, optional
        List of controllers that influence agent behavior. Default is None.
    logger : Logger, optional
        Logger instance for recording simulation data. Default is None.
    renderer : Renderer, optional
        Renderer instance for visualization. Default is None.

    Attributes
    ----------
    dt : float
        Simulation timestep, inherited from the integrator.
    T : float
        Total simulation time duration.
    populations : list of Population
        List of agent populations in the simulation.
    environment : Environment
        The simulation environment.
    interactions : list of Interaction or None
        List of interaction models.
    controllers : list of Controller or None
        List of control strategies.
    logger : Logger or None
        Data logger instance.
    renderer : Renderer or None
        Visualization renderer.
    integrator : Integrator
        Numerical integration scheme.

    Config Requirements
    -------------------
    The YAML configuration file must contain the following parameters under the simulator section:

    T : float, optional
        Total simulation time. Default is ``10.0``.
    dt : float, optional
        Simulation timestep. Default is ``0.01``.

    Notes
    -----
    The simulation loop executes the following steps in order:
    
    1. **Logging**: Record current simulation state
    2. **Rendering**: Visualize current state (if renderer is provided)
    3. **Control**: Computes control actions (if controllers are provided)
    4. **Interactions**: Computes inter-agent forces (if interactions are provided)
    5. **Integration**: Update agent states using numerical integration
    6. **Reset Forces**: Clear interaction forces for next timestep
    7. **Environment Update**: Update environment state

    Examples
    --------
    Basic simulation setup:

    .. code-block:: python

        from swarmsim import BrownianMotion, Simulator
        from swarmsim.Environments import EmptyEnvironment
        from swarmsim.Integrators import EulerMaruyama
        
        # Create components
        population = BrownianMotion('config.yaml')
        environment = EmptyEnvironment('config.yaml')
        integrator = EulerMaruyama('config.yaml')
        
        # Create and run simulator
        simulator = Simulator(
            config_path='config.yaml',
            populations=[population],
            environment=environment,
            integrator=integrator
        )
        simulator.simulate()

    Example YAML configuration:

    .. code-block:: yaml

        simulator:
            T: 50.0  # Run simulation for 50 time units
    """

    def __init__(self,
                 config_path: str,
                 populations: list[Population],
                 environment: Environment,
                 integrator: Integrator,
                 interactions: list[Interaction] | None =None,
                 controllers: list[Controller] | None =None,
                 logger: Logger | None =None,
                 renderer: Renderer | None =None,
                 ) -> None:
        """
        Initialize the Simulator with all required and optional components.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file containing simulation parameters.
        populations : list of Population
            List of agent populations to include in the simulation.
        environment : Environment
            The environment where agents operate.
        integrator : Integrator
            Numerical integration scheme for state evolution.
        interactions : list of Interaction, optional
            Inter-agent interaction models. Default is None.
        controllers : list of Controller, optional
            Control strategies for influencing agent behavior. Default is None.
        logger : Logger, optional
            Data recording component. Default is None.
        renderer : Renderer, optional
            Visualization component. Default is None.

        Raises
        ------
        FileNotFoundError
            If the configuration file cannot be found.
        """

        config: dict = load_config(config_path)

        simulator_config: dict = config.get('simulator', {})
        self.dt: float = integrator.dt
        self.T: float = simulator_config.get('T', 10)

        # get parameters from initialization
        self.populations: list[Population] = populations
        self.environment: Environment = environment
        self.interactions: list[Interaction] | None = interactions
        self.controllers: list[Controller] | None = controllers
        self.logger: Logger | None = logger
        self.renderer: Renderer | None = renderer
        self.integrator: Integrator = integrator

    def simulate(self):
        """
        Execute the main simulation loop.

        This method runs the complete simulation from t=0 to t=T, coordinating all
        simulation components in the proper sequence. It includes progress tracking
        and handles early termination if requested by the logger.

        The simulation loop performs the following steps at each timestep:
        
        1. **Data Logging**: Record current simulation state and check for termination
        2. **Visualization**: Render current state if renderer is available
        3. **Control Actions**: Apply control inputs from all controllers
        4. **Agent Interactions**: Compute forces between agents
        5. **State Integration**: Update agent positions using the integrator
        6. **Force Reset**: Clear interaction forces for the next timestep
        7. **Environment Update**: Update environmental conditions

        Notes
        -----
        - Progress is displayed using a progress bar
        - Controllers operate at their specified sampling rates
        - Interaction forces are reset at each timestep
        - Early termination is possible if the logger returns a done flag
        - All populations are reset to initial conditions before simulation starts

        Raises
        ------
        RuntimeError
            If any component fails during simulation execution.
        """

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

        for population in self.populations:
            population.reset()

        if self.interactions is not None:
            for interaction in self.interactions:
                interaction.reset()

        for t in range(num_steps):
            #print(t)
            done = self.logger.log()  # Log and get done flag

            # Render the Scene if a renderer is defined
            if self.renderer is not None:
                self.renderer.render()

            # Implement the control actions
            if self.controllers is not None:
                for c in self.controllers:
                    if c.dt is None or (t % (round(c.dt / self.dt))) == 0:
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
