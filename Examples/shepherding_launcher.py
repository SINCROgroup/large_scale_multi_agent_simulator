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

from swarmsim.Populations import BrownianMotion
from swarmsim.Populations import SimpleIntegrators
from swarmsim.Interactions import HarmonicRepulsion
from swarmsim.Controllers import ShepherdingLamaController
from swarmsim.Integrators import EulerMaruyamaIntegrator
from swarmsim.Renderers import ShepherdingRenderer
from swarmsim.Simulators import Simulator
from swarmsim.Environments import ShepherdingEnvironment
from swarmsim.Loggers import ShepherdingLogger
import pathlib

if __name__ == '__main__':
    
    config_path = str(pathlib.Path(__file__).resolve().parent.parent/"Configuration"/"shepherding_config.yaml")

    integrator = EulerMaruyamaIntegrator(config_path)

    environment = ShepherdingEnvironment(config_path)

    targets = BrownianMotion(config_path)
    herders = SimpleIntegrators(config_path)
    populations = [targets, herders]

    repulsion_ht = HarmonicRepulsion(targets, herders, config_path)
    interactions = [repulsion_ht]

    lamaController = ShepherdingLamaController(herders, targets, environment, config_path)
    controllers = [lamaController]

    renderer = ShepherdingRenderer(populations, environment, config_path)
    logger = ShepherdingLogger(populations, environment, config_path)

    simulator = Simulator(populations=populations, interactions=interactions, environment=environment, controllers=controllers,
                          integrator=integrator, logger=logger, renderer=renderer, config_path=config_path)

    simulator.simulate()
