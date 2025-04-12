import matplotlib.pyplot as plt
import numpy as np
import pygame
from swarmsim.Renderers import Renderer
from swarmsim.Environments import Environment


class BaseRenderer(Renderer):
    """
    A base renderer class responsible for rendering agents in an environment using either
    Matplotlib or Pygame.

    This class provides a flexible rendering system for visualizing populations within
    a simulation environment. It supports both **Matplotlib** (for static plots) and
    **Pygame** (for interactive rendering).

    Parameters
    ----------
    populations : list
        A list of population objects to render.
    environment : object, optional
        The environment instance in which the populations exist (default is None).
    config_path : str, optional
        Path to the YAML configuration file containing rendering settings (default is None).

    Attributes
    ----------
    populations : list
        List of population objects to be rendered.
    environment : object
        The environment in which agents operate.
    config : dict
        Dictionary containing rendering configuration parameters.
    agent_colors : str or list
        Color(s) used to render the agents.
    agent_shapes : str or list
        Shape(s) used to represent agents (e.g., `"circle"`, `"diamond"`).
    agent_sizes : float or list
        Size of the agents in the rendering.
    background_color : str
        Background color of the rendering window.
    render_mode : str
        Rendering mode, either `"matplotlib"` or `"pygame"`.
    render_dt : float
        Time delay between frames in seconds.
    fig, ax : matplotlib objects
        Matplotlib figure and axis (if using Matplotlib mode).
    window : pygame.Surface
        Pygame window surface (if using Pygame mode).
    clock : pygame.time.Clock
        Pygame clock for controlling frame rate.
    screen_size : tuple
        Size of the Pygame window.
    arena_size : tuple
        Size of the simulation arena in the Pygame window.

    Config requirements
    -------------------
    The YAML configuration file must contain a `renderer` section with the following parameters:

    agent_colors : str or list, optional
        Default color(s) for the agents (default is `"blue"`).
    agent_shapes : str or list, optional
        Shape(s) used to render agents (`"circle"` or `"diamond"`, default is `"circle"`).
    agent_sizes : float or list, optional
        Size of the agents (default is `1`).
    background_color : str, optional
        Background color of the rendering (default is `"white"`).
    render_mode : str, optional
        Rendering mode (`"matplotlib"` or `"pygame"`, default is `"matplotlib"`).
    render_dt : float, optional
        Time delay between frames in seconds (default is `0.05`).

    Notes
    -----
    - The rendering mode must be either `"matplotlib"` (for static plots) or `"pygame"` (for interactive rendering).
    - If using Matplotlib, a new figure is created at initialization.
    - If using Pygame, a window is created, and agents are rendered as circles or diamonds.

    Examples
    --------
    Example YAML configuration:

    .. code-block:: yaml

        renderer:
            agent_colors: ["red", "green"]
            agent_shapes: ["circle", "diamond"]
            agent_size: 5
            background_color: "black"
            render_mode: "pygame"
            render_dt: 0.1

    This configuration will render agents using **Pygame**, with red and green agents appearing
    as circles and diamonds, on a black background.
    """

    def __init__(self, populations: list, environment: Environment, config_path: str):
        """
        Initializes the renderer with the selected visualization mode.

        Parameters
        ----------
        populations : list
            A list of population objects to render.
        environment : object, optional
            The environment instance in which the populations exist (default is None).
        config_path : str, optional
            Path to the YAML configuration file (default is None).
        """
        super().__init__(populations, environment, config_path)

        # Load rendering settings from the config
        self.agent_colors = self.config.get('agent_colors', ['blue'])
        self.agent_shapes = self.config.get('agent_shapes', ['circle'])
        self.agent_sizes = self.config.get('agent_sizes', [1])

        self.background_color = self.config.get('background_color', 'white').lower()
        self.render_mode = self.config.get('render_mode', 'matplotlib').lower()
        self.render_dt = self.config.get('render_dt', 0.05)

        # Pygame setup
        self.window = None
        self.clock = None
        self.screen_size = (600, 600)
        self.arena_size = (600, 600)  # Smaller arena size
        self.scale_factor = None

        # Inside __init__ or called from it
        if self.render_mode == "matplotlib":
            self._setup_matplotlib()

    def _setup_matplotlib(self):
        """Initialize Matplotlib rendering."""
        arena_width, arena_height = self.environment.dimensions

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-arena_width / 2, arena_width / 2)
        self.ax.set_ylim(-arena_height / 2, arena_height / 2)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor(self.background_color)

        # Set labels and static elements once
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_title('Populations in the Environment')
        self.ax.grid(True)

        self.agent_scatters = []

        for population, color, shape, size in zip(self.populations,
                                                  self.agent_colors,
                                                  self.agent_shapes,
                                                  self.agent_sizes):
            marker = 'o' if shape == 'circle' else 'D'
            marker_size = size * 30  # Matplotlib marker size

            # Create empty scatter and cache it
            scatter = self.ax.scatter([], [], c=color, label=population.id,
                                      marker=marker, s=marker_size)
            self.agent_scatters.append(scatter)

        self.ax.legend(loc='upper right', framealpha=0.8)
        plt.tight_layout()

    def render(self):
        """
        Render the agents and environment using the selected renderer mode.

        Raises
        ------
        ValueError
            If an unsupported rendering mode is specified.
        """
        if self.render_mode == "matplotlib":
            return self.render_matplotlib()
        elif self.render_mode == "pygame":
            return self.render_pygame()
        else:
            raise ValueError("Unsupported renderer mode. Use 'matplotlib' or 'pygame'.")


    def render_matplotlib(self):
        """
        Efficient rendering using pre-created scatter plots.
        """
        # Update axis limits only if they change
        self.ax.set_xlim(-self.environment.dimensions[0] / 2, self.environment.dimensions[0] / 2)
        self.ax.set_ylim(-self.environment.dimensions[1] / 2, self.environment.dimensions[1] / 2)
        self.ax.set_facecolor(self.background_color)

        # Optional pre-hook
        self.pre_render_hook_matplotlib()

        # Update scatter plot data
        for scatter, population in zip(self.agent_scatters, self.populations):
            scatter.set_offsets(population.x)

        # Optional post-hook
        self.post_render_hook_matplotlib()

        # Only this is needed for refresh
        plt.pause(self.render_dt)

    def render_pygame(self):
        """
        Renders agents and the environment using Pygame.

        This method initializes a Pygame window (if not already created) and
        renders the agent populations using circles or diamond shapes.

        Notes
        -----
        - The function first fills the screen with the background color.
        - Calls `pre_render_hook_pygame()` before rendering agents.
        - Calls `post_render_hook_pygame()` after rendering.
        - Uses `pygame.display.flip()` to update the screen.
        - Uses `self.clock.tick(1 / self.render_dt)` to control rendering speed.

        Returns
        -------
        np.ndarray or pygame.Surface
            - If `render_mode == "rgb_array"`, returns a NumPy array representing the rendered frame.
            - Otherwise, returns the Pygame window.
        """
        # Initialize Pygame window if it hasn't been created
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(self.screen_size)
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Simulation Render")


            # Convert agent colors to Pygame format
            self.agent_colors = [
                pygame.Color(color) if isinstance(color, str) else pygame.Color(*color)
                for color in self.agent_colors
            ]

            # Calculate scale factor for rendering
            self.scale_factor = min(self.arena_size[0] / self.environment.dimensions[0],
                               self.arena_size[1] / self.environment.dimensions[1])

        # Fill the screen with the background color
        background_color = pygame.Color(self.background_color)
        self.window.fill(background_color)

        # Call the pre-render hook
        self.pre_render_hook_pygame()

        # Render agents
        for population, color, shape, size in zip(self.populations,
                                                  self.agent_colors,
                                                  self.agent_shapes,
                                                  self.agent_sizes):
            for position in population.x:
                # Convert simulation coordinates to screen coordinates
                x = int((position[0] + self.environment.dimensions[0] / 2) * self.scale_factor)
                y = int((self.environment.dimensions[1] / 2 - position[1]) * self.scale_factor)

                if shape == 'circle':
                    agent_radius = int(size / 2 * self.scale_factor)
                    pygame.draw.circle(self.window, color, (x, y), agent_radius)

                elif shape == 'diamond':
                    agent_side = int(size * np.sqrt(2) / 2 * self.scale_factor)
                    pygame.draw.polygon(self.window, color, [
                        (x, y - agent_side),  # Top
                        (x + agent_side, y),  # Right
                        (x, y + agent_side),  # Bottom
                        (x - agent_side, y),  # Left
                    ])

        # Call the post-render hook
        self.post_render_hook_pygame()

        # Update the display
        pygame.display.flip()

        # Control rendering speed
        self.clock.tick(1 / self.render_dt)

        # Return frame as an array if requested
        if self.render_mode != "rgb_array":
            frame = pygame.surfarray.array3d(self.window)
            frame = np.transpose(frame, (1, 0, 2))  # Convert to (height, width, channels)
            return frame

        return self.window

    def pre_render_hook_matplotlib(self):
        """
        Hook for adding custom pre-render logic in Matplotlib.

        This method is called before rendering agents, allowing subclasses
        to modify the figure (e.g., drawing additional elements).
        """
        pass

    def post_render_hook_matplotlib(self):
        """
        Hook for adding custom post-render logic in Matplotlib.

        This method is called after rendering agents, allowing subclasses
        to modify the figure (e.g., adding annotations).
        """
        pass

    def pre_render_hook_pygame(self):
        """
        Hook for adding custom pre-render logic in Pygame.

        This method is called before rendering agents, allowing subclasses
        to modify the display (e.g., drawing additional elements).
        """
        pass

    def post_render_hook_pygame(self):
        """
        Hook for adding custom post-render logic in Pygame.

        This method is called after rendering agents, allowing subclasses
        to modify the display (e.g., adding annotations).
        """
        pass

    def close(self):
        """
        Closes the Pygame window if it is open.

        This method ensures that Pygame is properly shut down, freeing
        up any allocated resources.

        Notes
        -----
        - If the window is not `None`, the function quits Pygame and resets the window to `None`.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
