"""

The `controllers` module provides both an interface to implement custom controllers and some examples of controllers.

Available Controllers
~~~~~~~~~~~~~~~~~~~~~

`Controller` : Abstract base class for defining custom controllers.
`ShepherdingLamaController` : A controller that guides the population to move behind the most distant target.
`GaussianRepulsion` : A controller that implements radial repulsion force whose intensity is shaped like a gaussian profile of null mean a Identity sandard deviation
`LightPattern` : A controller that projects a light pattern on the environment.

Usage
-----
To use a controller, import it and instantiate with a configuration file:

.. code-block:: python

    from swarmsim.controller import ShepherdingLamaController
    lamaController = ShepherdingLamaController(herders, targets, environment, config_path)
    controllers = [lamaController]

Modules
-------
- `base_controller` : Defines the abstract `controller` class, which serves as an interface for implementing custom control laws.
- `sheperding_lama_controller` : Implements `ShepherdingLamaController`, which is tasked with solving a sheperding problem.
- `spatial_inputs` : Implements `GaussianRepulsion` and `LightPattern`, which are control spatially heterogeneous static control laws.

Examples
--------
Example YAML configuration:

    .. code-block:: yaml

        LightPattern:
            pattern_path: ../Configuration/Config_data/BCL.jpeg
            dt: 0.1

    This defines a `LightPattern` that project the content of BCL.jpeg over the environment and that updates the control applied to the agents every 0.1 temporal siumlation units.

"""

from swarmsim.Controllers.base_controller import Controller
from swarmsim.Controllers.shepherding_lama_controller import ShepherdingLamaController
from swarmsim.Controllers.spatial_inputs import GaussianRepulsion
