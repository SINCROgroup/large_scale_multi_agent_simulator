"""
Controllers Module
==================

This module provides interfaces and implementations for controlling agent behavior in multi-agent simulations.
Controllers compute control actions that influence agent dynamics based on current states and objectives.

Available Controllers
---------------------

Controller
    Abstract base class defining the interface for all controller types.

ShepherdingLamaController
    Shepherding controller that guides one population using another population as herders.
    Implements the LAMA (Large Area Multi-agent) shepherding algorithm.

GaussianRepulsion
    Implements spatially-varying repulsion forces with Gaussian intensity profiles.

LightPattern
    Projects spatial light patterns from image files onto the environment for phototaxis control.

Temporal_pulses
    Generates temporally-varying pulse signals with periodic on/off patterns.

RectangularFeedback
    Provides binary feedback based on agent orientation relative to radial direction.

AngularFeedback
    Creates dynamic visual feedback using angular light patterns behind agents.

Key Features
------------
- **Modular Design**: Easy to implement custom control strategies
- **Multi-Population Support**: Controllers can coordinate multiple populations
- **Spatial Control**: Support for spatially-varying control fields
- **Configurable Timing**: Controllers can operate at different sampling rates

Examples
--------
Basic controller usage:

.. code-block:: python

    from swarmsim.Controllers import ShepherdingLamaController
    from swarmsim.Populations import BrownianMotion
    from swarmsim.Environments import EmptyEnvironment
    
    # Create populations and environment
    herders = BrownianMotion('config.yaml', 'Herders')
    targets = BrownianMotion('config.yaml', 'Targets')
    environment = EmptyEnvironment('config.yaml')
    
    # Create controller
    controller = ShepherdingLamaController(
        population=targets,
        environment=environment,
        config_path='config.yaml',
        other_populations=[herders]
    )
    
    # Get control action
    control_input = controller.get_action()

Configuration Examples
----------------------

Shepherding controller:

.. code-block:: yaml

    ShepherdingLamaController:
        dt: 0.1
        gain: 2.0
        target_radius: 5.0

Light pattern controller:

.. code-block:: yaml

    LightPattern:
        pattern_path: ../Configuration/Config_data/BCL.jpeg
        dt: 0.1

This defines a controller that projects the content of BCL.jpeg over the environment
and updates the control applied to agents every 0.1 simulation time units.

Modules
-------
- `base_controller`: Defines the abstract Controller class interface
- `shepherding_lama_controller`: Implements shepherding strategies
- `spatial_inputs`: Implements spatially heterogeneous control laws
"""

from swarmsim.Controllers.base_controller import Controller
from swarmsim.Controllers.shepherding_lama_controller import ShepherdingLamaController
from swarmsim.Controllers.spatial_inputs import (
    GaussianRepulsion,
    LightPattern,
    Temporal_pulses,
    AngularFeedback
)
