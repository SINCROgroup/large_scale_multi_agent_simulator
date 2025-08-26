"""
Integrators Module
==================

This module provides numerical integration schemes for evolving agent dynamics in multi-agent simulations.
Integrators handle both deterministic and stochastic differential equations, enabling realistic agent motion
with noise and uncertainty.

Available Integrators
---------------------

Integrator
    Abstract base class defining the interface for all numerical integration methods.

EulerMaruyama
    Stochastic integrator for SDEs using the Euler-Maruyama scheme. Suitable for
    populations with both drift and diffusion components (e.g., Brownian motion).

Key Features
------------
- **Stochastic Integration**: Support for stochastic differential equations (SDEs)
- **Modular Design**: Easy to implement custom integration schemes
- **Population Support**: Handles multiple populations with different dynamics
- **Configurable Timestep**: Adjustable integration timestep for accuracy vs. speed

Mathematical Background
-----------------------
The integrators solve equations of the form:

    dx = f(x, t) dt + g(x, t) dW

where:
- f(x, t) is the drift term (deterministic component)
- g(x, t) is the diffusion term (stochastic component)  
- dW is a Wiener process (Brownian motion)

Examples
--------
Basic integrator usage:

.. code-block:: python

    from swarmsim.Integrators import EulerMaruyama
    from swarmsim.Populations import BrownianMotion
    
    # Create integrator
    integrator = EulerMaruyama('config.yaml')
    
    # Create population
    population = BrownianMotion('config.yaml')
    population.reset()
    
    # Perform integration step
    integrator.step([population])

Configuration Example
---------------------

.. code-block:: yaml

    integrator:
        dt: 0.01  # Small timestep for accuracy

For stochastic systems, smaller timesteps generally improve accuracy but increase
computational cost.
"""

# Import key integrator classes for easy access
from .base_integrator import Integrator
from .euler_maruyama import EulerMaruyamaIntegrator
