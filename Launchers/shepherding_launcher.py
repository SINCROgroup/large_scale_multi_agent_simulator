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
from Populations.simple_integrators import SimpleIntegrators
from Interactions.harmonic_repulsion import HarmonicRepulsion
from Controllers.shepherding_lama_controller import ShepherdingLamaController
from Integrators.euler_maruyama import EulerMaruyamaIntegrator
from Renderers.base_renderer import BaseRenderer
from Simulators.base_simulator import Simulator
from Environments.empty_environment import EmptyEnvironment
from Loggers.base_logger import BaseLogger

if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), '..', 'shepherding_config.yaml')

    integrator = EulerMaruyamaIntegrator(config_path)

    environment = EmptyEnvironment(config_path)

    targets = BrownianMotion(config_path)
    herders = SimpleIntegrators(config_path)
    populations = [targets, herders]

    repulsion_ht = HarmonicRepulsion(targets, herders, config_path)
    interactions = [repulsion_ht]

    lamaController = ShepherdingLamaController(herders, targets, environment, config_path)
    controllers = [lamaController]

    renderer = BaseRenderer(populations, environment, config_path)
    logger = BaseLogger(config_path)

    simulator = Simulator(populations=populations, interactions=interactions, environment=environment, controllers=controllers,
                          integrator=integrator, logger=logger, renderer=renderer, config_path=config_path)

    simulator.simulate()
