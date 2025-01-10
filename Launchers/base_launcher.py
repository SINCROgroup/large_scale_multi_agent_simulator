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

from Interactions.harmonic_repulsion import HarmonicRepulsion
from Integrators.euler_maruyama import EulerMaruyamaIntegrator

from Controllers.spatial_inputs import Gaussian_Repulsion

from Renderers.base_renderer import BaseRenderer

from Simulators.base_simulator import Simulator

from Environments.empty_environment import EmptyEnvironment

from Loggers.base_logger import BaseLogger


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), '../Configs', 'base_config.yaml')

    integrator = EulerMaruyamaIntegrator(config_path)
    
    environment = EmptyEnvironment(config_path)
    
    population1 = BrownianMotion(config_path)
    population2 = FixedPopulation(config_path)
    populations = [population1, population2]
    controllers = []

    controller = Gaussian_Repulsion(population2,environment,config_path)
    controllers =[controller]

    repulsion_12 = HarmonicRepulsion(population1, population2, config_path)
    interactions = [repulsion_12]

    renderer = BaseRenderer(populations, environment, config_path)
    logger = BaseLogger(populations, environment, config_path)

    simulator = Simulator(populations=populations, interactions=interactions, environment=environment, controllers=controllers,
                          integrator=integrator, logger=logger, renderer=renderer, config_path=config_path)

    simulator.simulate()
