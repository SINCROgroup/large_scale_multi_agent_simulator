"""
Population Module
=================

This module contains implementations of various agent population types for multi-agent simulations.
Each population type defines different dynamics and behaviors for groups of agents.

Available Population Types
--------------------------

Population
    Abstract base class defining the interface for all population types.

BrownianMotion
    Implements biased Brownian motion with drift and diffusion components.

FixedPopulation
    Represents stationary agents with fixed positions.

SimpleIntegrators
    Single integrator dynamics (velocity-controlled agents).

DampedDoubleIntegrators
    Double integrator dynamics (position and velocity states, controlled in acceleration).

LightSensitive_PTW
    Persistent turning walker model with light sensitivity for biological agents.

Examples
--------
Basic usage of a Brownian motion population:

.. code-block:: python

    from swarmsim.Populations import BrownianMotion
    
    # Create population from configuration
    population = BrownianMotion('config.yaml')
    
    # Reset to initial conditions
    population.reset()
    
    # Access current state
    positions = population.x
    
    # Get dynamics components
    drift = population.get_drift()
    diffusion = population.get_diffusion()
"""

from swarmsim.Populations.population import Population
from swarmsim.Populations.brownian_motion import BrownianMotion
from swarmsim.Populations.fixed_population import FixedPopulation
from swarmsim.Populations.simple_integrators import SimpleIntegrators
from swarmsim.Populations.double_integrators import DampedDoubleIntegrators
from swarmsim.Populations.parsistent_turning_walker import LightSensitive_PTW
