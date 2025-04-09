# This is the base launcher for LS_MAS (Large scale - Multi agent Simulator)
# Authors 
# 
#
#
#
# Description:
#

#TODO: Test Loading from file 


import sys
import os

from swarmsim.Loggers.position_logger import PositionLogger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import swarmsim as ss

from swarmsim.Populations import BrownianMotion
from swarmsim.Populations import FixedPopulation

from swarmsim.Interactions import HarmonicRepulsion
from swarmsim.Integrators import EulerMaruyamaIntegrator

from swarmsim.Controllers import GaussianRepulsion

from swarmsim.Renderers import BaseRenderer

from swarmsim.Simulators import Simulator

from swarmsim.Environments import EmptyEnvironment

from swarmsim.Loggers import BaseLogger

import pathlib


if __name__ == '__main__':

    config_path = str(pathlib.Path(__file__).resolve().parent.parent/"Configuration"/"base_config.yaml")

    integrator = EulerMaruyamaIntegrator(config_path)
    
    environment = EmptyEnvironment(config_path)
    
    population1 = BrownianMotion(config_path)
    population2 = FixedPopulation(config_path)
    populations = [population1, population2]
    controllers = []

    controller = GaussianRepulsion(population2, environment, config_path)
    controllers =[controller]

    repulsion_12 = HarmonicRepulsion(population1, population2, config_path)
    interactions = [repulsion_12]

    renderer = BaseRenderer(populations, environment, config_path)
    logger = PositionLogger(populations, environment, config_path)

    simulator = Simulator(populations=populations, interactions=interactions, environment=environment, controllers=controllers,
                          integrator=integrator, logger=logger, renderer=renderer, config_path=config_path)

    simulator.simulate()
