"""

The `Integrators` module provides numerical integration methods for simulating
the movement of agents.

Classes
-------
- `Integrator`:
    - Abstract base class for numerical integration.
    - Requires implementation of the `step` method.
- `EulerMaruyamaIntegrator`:
    - Stochastic integration method that updates agent positions based on drift and diffusion terms.

Usage
-----
To use an integrator, import it and instantiate with a configuration file:

.. code-block:: python

    from swarmsim.integrators import EulerMaruyamaIntegrator
    integrator = EulerMaruyamaIntegrator(config_path="config.yaml")
    integrator.step(populations)


This will perform a **single integration step** using the Euler-Maruyama method.

"""

# Import key integrator classes for easy access
from .base_integrator import Integrator
from .euler_maruyama import EulerMaruyamaIntegrator
