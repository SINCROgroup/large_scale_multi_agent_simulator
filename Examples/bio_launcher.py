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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import swarmsim as ss

from swarmsim.Populations.parsistent_turning_walker import LightSensitive_PTW

from swarmsim.Interactions import HarmonicRepulsion
from swarmsim.Integrators import EulerMaruyamaIntegrator

from swarmsim.Controllers.spatial_inputs import LightPattern

from swarmsim.Renderers import BaseRenderer

from swarmsim.Simulators import Simulator

from swarmsim.Environments import EmptyEnvironment

from swarmsim.Loggers import BaseLogger

import pathlib


if __name__ == '__main__':

    config_path = str(pathlib.Path(__file__).resolve().parent.parent/"Configuration"/"bio_config.yaml")

    integrator = EulerMaruyamaIntegrator(config_path)
    
    environment = EmptyEnvironment(config_path)
    
    population1 = LightSensitive_PTW(config_path)
    populations = [population1]
    controllers = []

    controller = LightPattern(population1,environment,config_path) 
    controllers =[controller]

    interactions = []

    renderer = BaseRenderer(populations, environment, config_path)
    logger = BaseLogger(populations, environment, config_path)

    simulator = Simulator(populations=populations, interactions=interactions, environment=environment, controllers=controllers,
                          integrator=integrator, logger=logger, renderer=renderer, config_path=config_path)

    simulator.simulate()
