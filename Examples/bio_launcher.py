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

from swarmsim.Controllers.spatial_inputs import LightPattern, Temporal_pulses

from swarmsim.Renderers.bio_renderer import BioRenderer

from swarmsim.Simulators import Simulator

from swarmsim.Environments import EmptyEnvironment

from swarmsim.Loggers import PositionLogger

import pathlib


if __name__ == '__main__':

    config_path = str(pathlib.Path(__file__).resolve().parent.parent/"Configuration"/"bio_config.yaml")

    integrator = EulerMaruyamaIntegrator(config_path)
    
    environment = EmptyEnvironment(config_path)
    
    population1 = LightSensitive_PTW(config_path)
    populations = [population1]

    #controller = LightPattern(population1, environment, config_path)
    controller = Temporal_pulses(population1, environment, config_path)

    controllers = [controller]

    interactions = []

    renderer = None #BioRenderer(populations, environment, config_path, controller)
    logger = PositionLogger(populations, environment, config_path)

    simulator = Simulator(populations=populations, environment=environment, controllers=controllers,
                          integrator=integrator, logger=logger, renderer=renderer, config_path=config_path)

    simulator.simulate()
