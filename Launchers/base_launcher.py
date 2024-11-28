# This is the base launcher for LS_MAS (Large scale - Multi agent Simulator)
# Authors 
# 
#
#
#
# Description:
#

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from Populations.brownian_motion import BrownianMotion
from Populations.fixed_population import FixedPopulation
from Interactions.constant_repulsion import RepulsionConst
from Integrators.euler_maruyama import EulerMaruyamaIntegrator
from Renderers.renderer import Renderer
from Simulators.base_simulator import Simulator
from Environments.empty_environment import EmptyEnvironment
from Loggers.base_logger import Logger


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')

    integrator = EulerMaruyamaIntegrator(config_path)

    renderer = Renderer(config_path)
    
    environment = EmptyEnvironment(config_path)
    
    population1 = BrownianMotion(config_path)
    population2 = FixedPopulation(config_path)
    populations = [population1, population2]

    repulsion_12 = RepulsionConst(population1, population2, config_path)
    interactions = [repulsion_12]

    logger = Logger(config_path)

    simulator = Simulator(populations=populations, interactions=interactions, environment=environment, controllers=None,
                          integrator=integrator, logger=logger, renderer=renderer, config_path=config_path)

    simulator.simulate()
