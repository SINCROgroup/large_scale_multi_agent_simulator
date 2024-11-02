import matplotlib.pyplot as plt
import yaml
import pygame
import numpy as np


class Render:
    """
    A class responsible for rendering the agents in the environment.
    """

    def __init__(self, config_path):
        """
        Initializes the Render class with configuration parameters from a YAML file.

        Args:
            config_path (str): The path to the YAML configuration file.
        """
        # Load config params from YAML file
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

        render_config = config.get('render', {})
        self.agent_color = render_config.get('agent_color', 'blue')
        self.background_color = render_config.get('background_color', 'white')
        self.render_mode = render_config.get('render_mode', 'matplotlib')
        self.render_dt = render_config.get('render_dt', 0.05)  # Default to 0.05 seconds if not specified

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
            self.ax.set_title('Agents in the Environment')
            self.ax.grid(True)

    def render(self, agents, environment):
        """
        Renders the agents in the given environment using the selected render mode.

        Args:
            agents: An object containing the positions of the agents as `agents.x`.
            environment: An instance of the environment class, which provides the dimensions of the environment.
        """
        if self.render_mode == "matplotlib":
            self._render_matplotlib(agents, environment)
        elif self.render_mode == "pygame":
            self._render_pygame(agents, environment)
        else:
            raise ValueError("Unsupported render mode. Use 'matplotlib' or 'pygame'.")

    def _render_matplotlib(self, agents, environment):
        """
        Renders the agents in the given environment using Matplotlib.

        Args:
            agents: An object containing the positions of the agents as `agents.x`.
            environment: An instance of the environment class, which provides the dimensions of the environment.
        """
        self.ax.clear()
        self.ax.set_xlim(-environment.dimensions[0] / 2, environment.dimensions[0] / 2)
        self.ax.set_ylim(-environment.dimensions[1] / 2, environment.dimensions[1] / 2)
        self.ax.set_aspect('equal')  # Ensure equal scaling for x and y axes
        self.ax.set_facecolor(self.background_color)

        # Plot agents
        self.ax.scatter(agents.x[:, 0], agents.x[:, 1], c=self.agent_color, label='Agents')

        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_title('Agents in the Environment')
        self.ax.legend()
        self.ax.grid(True)

        plt.pause(self.render_dt)

    def _render_pygame(self, agents, environment):
        """
        Renders the agents in the given environment using Pygame, similar to the OpenAI Gym rendering.

        Args:
            agents: An object containing the positions of the agents as `agents.x`.
            environment: An instance of the environment class, which provides the dimensions of the environment.
        """
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Agents in the Environment")
            self.clock = pygame.time.Clock()

        agent_color = pygame.Color(self.agent_color)
        background_color = pygame.Color(self.background_color)

        # Scale factors to convert environment coordinates to arena coordinates
        scale = min(self.arena_size[0] / environment.dimensions[0], self.arena_size[1] / environment.dimensions[1])

        # Handle Pygame events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        # Clear the screen
        self.window.fill(background_color)

        # Draw grid lines inside the arena
        grid_color = (200, 200, 200)  # Light gray
        for x in range(0, self.arena_size[0], int(self.arena_size[0] / 10)):
            pygame.draw.line(self.window, grid_color, (x + 100, 100), (x + 100, 700))
        for y in range(0, self.arena_size[1], int(self.arena_size[1] / 10)):
            pygame.draw.line(self.window, grid_color, (100, y + 100), (700, y + 100))

        # Draw x and y ticks and labels
        font = pygame.font.SysFont('Arial', 16)
        for i in range(0, 11):
            # X-axis ticks
            x = 100 + i * (self.arena_size[0] / 10)
            tick_label = f"{int(-environment.dimensions[0] / 2 + i * (environment.dimensions[0] / 10))}"
            tick_text = font.render(tick_label, True, (0, 0, 0))
            self.window.blit(tick_text, (x - 10, 710))
            # Y-axis ticks
            y = 100 + i * (self.arena_size[1] / 10)
            tick_label = f"{int(environment.dimensions[1] / 2 - i * (environment.dimensions[1] / 10))}"
            tick_text = font.render(tick_label, True, (0, 0, 0))
            self.window.blit(tick_text, (60, y - 10))

        # Draw agents within the arena
        for position in agents.x:
            x = int((position[0] + environment.dimensions[0] / 2) * scale) + 100
            y = int((environment.dimensions[1] / 2 - position[1]) * scale) + 100
            pygame.draw.circle(self.window, agent_color, (x, y), 5)

        # Draw environment boundaries (arena boundaries)
        boundary_color = (0, 0, 0)  # Black
        pygame.draw.rect(self.window, boundary_color, (100, 100, self.arena_size[0], self.arena_size[1]), 2)

        # Draw legend
        legend_text = font.render('Agents', True, agent_color)
        self.window.blit(legend_text, (10, 10))

        # Draw axis labels
        x_label = font.render('X Position', True, (0, 0, 0))
        y_label = font.render('Y Position', True, (0, 0, 0))
        self.window.blit(x_label, (self.screen_size[0] // 2 - 50, 750))
        self.window.blit(y_label, (10, self.screen_size[1] // 2 - 50))

        # Update the display
        pygame.display.flip()
        self.clock.tick(1 / self.render_dt)

    def close(self):
        """
        Closes the Pygame window if it is open.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
