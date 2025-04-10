import sys
import os
import pathlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from swarmsim.Populations import BrownianMotion
from swarmsim.Populations import FixedPopulation
from swarmsim.Interactions import HarmonicRepulsion
from swarmsim.Integrators import EulerMaruyamaIntegrator
from swarmsim.Renderers import BaseRenderer
from swarmsim.Loggers.position_logger import PositionLogger
from swarmsim.Simulators import Simulator
from swarmsim.Environments import EmptyEnvironment

if __name__ == '__main__':

    config_path = str(pathlib.Path(__file__).resolve().parent.parent/"Configuration"/"base_config.yaml")

    population1 = BrownianMotion(config_path)
    population2 = FixedPopulation(config_path)
    populations = [population1, population2]

    environment = EmptyEnvironment(config_path)
    repulsion_12 = HarmonicRepulsion(population1, population2, config_path)
    interactions = [repulsion_12]

    integrator = EulerMaruyamaIntegrator(config_path)
    controllers = []

    renderer = BaseRenderer(populations, environment, config_path)
    logger = PositionLogger(populations, environment, config_path)

    simulator = Simulator(populations=populations, interactions=interactions, environment=environment, controllers=controllers,
                          integrator=integrator, logger=logger, renderer=renderer, config_path=config_path)

    simulator.simulate()
