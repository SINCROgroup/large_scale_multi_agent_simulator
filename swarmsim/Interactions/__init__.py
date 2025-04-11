"""
Interactions Module
===================

The `interactions` module defines different types of interactions between agents in
multi-agent shepherding simulations. These interactions influence agent behaviors
by applying forces such as repulsion or attraction.

Available Interactions
----------------------

- `Interaction` : Abstract base class that defines the interface for all interaction models.
- `HarmonicRepulsion` : Implements a finite-range harmonic repulsion force.
- `PowerLawRepulsion` : Implements a power-law repulsion force, where force decays with distance.

Usage
-----

To use an interaction model, import it and instantiate with a configuration file:

.. code-block:: python

    from swarmsim.interactions import HarmonicRepulsion
    interaction = HarmonicRepulsion(target_population, source_population, config="config.yaml")
    forces = interaction.get\_interaction()


Examples
--------

Example YAML configuration:

.. code-block:: yaml

    harmonic_repulsion:
        strength: 1.5
        max_distance: 10.0

    power_law_repulsion:
        strength: 2.0
        max_distance: 5.0
        p: 3.0


This sets up two different repulsion models with distinct behaviors.

"""

from .interaction import Interaction
from .harmonic_repulsion import HarmonicRepulsion
from .power_law_repulsion import PowerLawRepulsion
from .power_law_interaction import PowerLawInteraction

__all__ = ["Interaction", "HarmonicRepulsion", "PowerLawRepulsion", "PowerLawInteraction"]
