"""
Environments Module
===================

The `environments` module provides different types of environments for multi-agent
swarm simulations. These environments define the physical space in which agents operate
and interact with their surroundings.

Available Environments
----------------------
- `Environment` : Abstract base class for defining environments.
- `EmptyEnvironment` : A static environment with no external forces acting on agents.
- `ShepherdingEnvironment` : A dynamic environment where the goal moves along a linear path.

Usage
-----
To use an environment, import it and instantiate with a configuration file:

.. code-block:: python

    from swarmsim.environments import ShepherdingEnvironment
    env = ShepherdingEnvironment(config_path="config.yaml")
    env.update()

Modules
-------
- `base_environment` : Defines the abstract `Environment` class, which serves as a base for all environments.
- `empty_environment` : Implements `EmptyEnvironment`, which has no external forces.
- `shepherding_environment` : Implements `ShepherdingEnvironment`, where the goal moves dynamically.

Examples
--------
Example YAML configuration:

.. code-block:: yaml

    environment:
        dimensions: [100, 100]
        goal_radius: 5
        goal_pos: [0, 0]
        final_goal_pos: [20, -20]
        num_steps: 2000

This will create an environment where the goal moves from (0, 0) to (20, -20) over 2000 steps.

"""

from .base_environment import Environment
from .empty_environment import EmptyEnvironment
from .shepherding_environment import ShepherdingEnvironment

__all__ = ["Environment", "EmptyEnvironment", "ShepherdingEnvironment"]
