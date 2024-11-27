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


from Models.brownian_motion import BrownianMotion
from Models.fixed_population import Fixed_population
from Models.constant_repulsion import Repulsion_const
from Integrators.euler_maruyama import EulerMaruyamaIntegrator
from Renders.render import Render
from Simulators.base_simulator import Simulator
from Environments.empty_environment import EmptyEnvironment
from Loggers.base_logger import  Logger


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')

    integrator = EulerMaruyamaIntegrator(config_path)

    render = Render(config_path)
    
    environment = EmptyEnvironment(config_path)
    
    population1 = BrownianMotion(config_path)
    population2 = Fixed_population(config_path)
    populations = [population1,population2]

    repulsion_12 = Repulsion_const(population1,population2,config_path)
    interactions = [repulsion_12]

    logger = Logger(config_path)

    simulator = Simulator(populations=populations,
                          environment=environment,
                          interactions=interactions,
                          controller=None,
                          integrator=integrator,
                          logger=logger,
                          render=render,
                          config_path=config_path)

    simulator.simulate()
