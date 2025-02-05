"""
swarmsim.Renderers
==================

This module provides rendering utilities for visualizing multi-agent environments.

It includes:
    - A **base renderer** (`BaseRenderer`) supporting both **Matplotlib** and **Pygame**.
    - A **specialized renderer** (`ShepherdingRenderer`) for shepherding simulations, which includes a **goal region visualization**.
    - An **abstract interface** (`Renderer`) to define custom rendering implementations.

Module Contents
---------------
    - `Renderer` : Abstract base class defining the rendering interface.
    - `BaseRenderer` : General-purpose renderer supporting Matplotlib and Pygame.
    - `ShepherdingRenderer` : Specialized renderer with goal visualization.

Examples
--------
Example usage:

.. code-block:: python

    from swarmsim.Renderers import BaseRenderer

    renderer = BaseRenderer(populations, environment, config_path="config.yaml")
    renderer.render()



Notes
-----
    - If `render_mode="matplotlib"`, the renderer creates a static figure with a pause interval.
    - If `render_mode="pygame"`, a Pygame window is created for interactive rendering.
    - The **ShepherdingRenderer** adds a visualized goal region for shepherding tasks.

"""

from swarmsim.Renderers.renderer import Renderer
from swarmsim.Renderers.base_renderer import BaseRenderer
from swarmsim.Renderers.shepherding_renderer import ShepherdingRenderer
