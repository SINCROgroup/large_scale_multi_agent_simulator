import matplotlib.pyplot as plt
import numpy as np
import pygame
from swarmsim.Renderers import Renderer


class BaseRenderer(Renderer):
    """
    A base renderer class responsible for rendering the agents in the environment.
    """

    def __init__(self, populations, environment=None, config_path=None):
        super().__init__(populations, environment, config_path)

        self.agent_colors = self.config.get('agent_colors', 'blue')
        self.agent_shapes = self.config.get('agent_shapes', 'circle')
        self.agent_size = self.config.get('agent_size', 1)

        self.background_color = self.config.get('background_color', 'white')
        self.render_mode = self.config.get('render_mode', 'matplotlib')
        self.render_dt = self.config.get('render_dt', 0.05)  # Default to 0.05 seconds if not specified

        # Pygame window setup
        self.window = None
        self.clock = None
        self.screen_size = (800, 800)
        self.arena_size = (600, 600)  # Arena smaller than the window

        # Matplotlib setup
        self.fig, self.ax = None, None
        if self.render_mode == "matplotlib":
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(-100 / 2, 100 / 2)  # Default limits, can be updated later
            self.ax.set_ylim(-100 / 2, 100 / 2)
            self.ax.set_aspect('equal')  # Ensure equal scaling for x and y axes
            self.ax.set_facecolor(self.background_color)
            self.ax.set_xlabel('X Position')
            self.ax.set_ylabel('Y Position')
            self.ax.set_title('Populations in the Environment')
            self.ax.grid(True)

    def render(self):
        """
        Render the agents and environment using the selected renderer mode.
        """
        if self.render_mode == "matplotlib":
            self._render_matplotlib()
        elif self.render_mode == "pygame":
            self._render_pygame()
        else:
            raise ValueError("Unsupported renderer mode. Use 'matplotlib' or 'pygame'.")

    def _render_matplotlib(self):
        """
        Render agents and environment using Matplotlib.
        """
        self.ax.clear()
        self.ax.set_xlim(-self.environment.dimensions[0] / 2, self.environment.dimensions[0] / 2)
        self.ax.set_ylim(-self.environment.dimensions[1] / 2, self.environment.dimensions[1] / 2)
        self.ax.set_aspect('equal')  # Ensure equal scaling for x and y axes
        self.ax.set_facecolor(self.background_color)

        # Call the pre-render hook
        self.pre_render_hook_matplotlib()

        # Plot agents
        for i, (population, color, shape, size) in enumerate(zip(self.populations,
                                                                 self.agent_colors,
                                                                 self.agent_shapes,
                                                                 self.agent_size)):

            # Adjust size scaling for scatter plot
            marker_size = (size / 2) ** 2  # Matplotlib `s` is proportional to the area of the marker
            marker_size = size * 30

            if shape == 'circle':
                self.ax.scatter(population.x[:, 0], population.x[:, 1],
                                c=color, label=population.id,
                                marker='o', s=marker_size)

            elif shape == 'diamond':
                self.ax.scatter(population.x[:, 0], population.x[:, 1],
                                c=color, label=population.id,
                                marker='D', s=marker_size)  # 'D' is the diamond marker in matplotlib

        # Call the post-render hook
        self.post_render_hook_matplotlib()

        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')

        self.ax.set_title('Populations in the Environment')
        self.ax.legend()
        self.ax.grid(True)

        plt.pause(self.render_dt)

    def _render_pygame(self):
        """
        Render agents and environment using Pygame.
        """
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Populations in the Environment")
            self.clock = pygame.time.Clock()

        background_color = pygame.Color(self.background_color)
        self.window.fill(background_color)

        # Call the pre-render hook
        self.pre_render_hook_pygame()

        # Render agents
        agent_colors = [
            pygame.Color(color) if isinstance(color, str) else pygame.Color(*color)
            for color in self.agent_colors
        ]
        scale = min(self.arena_size[0] / self.environment.dimensions[0],
                    self.arena_size[1] / self.environment.dimensions[1])
        for i, (population, color, shape, size) in enumerate(zip(self.populations,
                                                                 agent_colors,
                                                                 self.agent_shapes,
                                                                 self.agent_size)):
            for position in population.x:
                x = int((position[0] + self.environment.dimensions[0] / 2) * scale) + 100
                y = int((self.environment.dimensions[1] / 2 - position[1]) * scale) + 100

                if shape == 'circle':
                    agent_radius = int(size / 2 * scale)
                    pygame.draw.circle(self.window, color, (x, y), agent_radius)

                if shape == 'diamond':
                    agent_side = int(size * np.sqrt(2) / 2 * scale)
                    pygame.draw.polygon(self.window, color, [
                        (x, y - agent_side),  # Top
                        (x + agent_side, y),  # Right
                        (x, y + agent_side),  # Bottom
                        (x - agent_side, y),  # Left
                    ])

        # Call the post-render hook
        self.post_render_hook_pygame()

        pygame.display.flip()
        self.clock.tick(1 / self.render_dt)

    def pre_render_hook_matplotlib(self):
        """Hook for adding custom pre-render logic for Matplotlib."""
        pass

    def post_render_hook_matplotlib(self):
        """Hook for adding custom post-render logic for Matplotlib."""
        pass

    def pre_render_hook_pygame(self):
        """Hook for adding custom pre-render logic for Pygame."""
        pass

    def post_render_hook_pygame(self):
        """Hook for adding custom post-render logic for Pygame."""
        pass

    def close(self):
        """
        Closes the Pygame window if it is open.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
