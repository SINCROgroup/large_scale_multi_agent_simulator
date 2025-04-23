import sys
import os
import pathlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from swarmsim.Populations import BrownianMotion
from swarmsim.Interactions import LennardJones
from swarmsim.Integrators import EulerMaruyamaIntegrator
from swarmsim.Renderers import BaseRenderer
from swarmsim.Loggers import BaseLogger
from swarmsim.Simulators import Simulator
from swarmsim.Environments import EmptyEnvironment

if __name__ == '__main__':

    config_path = str(pathlib.Path(__file__).resolve().parent.parent/"Configuration"/"single_config.yaml")

    population = BrownianMotion(config_path)

    populations = [population]

    environment = EmptyEnvironment(config_path)
    interaction = LennardJones(population, population, config_path, "Interaction")
    interactions = [interaction]

    integrator = EulerMaruyamaIntegrator(config_path)

    renderer = BaseRenderer(populations, environment, config_path)
    logger = BaseLogger(populations, environment, config_path)

    simulator = Simulator(populations=populations, interactions=interactions, environment=environment,
                          integrator=integrator, logger=logger, renderer=renderer, config_path=config_path)

    simulator.simulate()
