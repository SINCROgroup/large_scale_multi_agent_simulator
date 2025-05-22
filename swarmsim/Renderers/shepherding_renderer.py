import matplotlib.pyplot as plt
import pygame
from swarmsim.Renderers import BaseRenderer
from swarmsim.Environments import Environment


class ShepherdingRenderer(BaseRenderer):
    """
    A renderer that extends `BaseRenderer` to include visualization of a goal region
    for shepherding tasks.

    This renderer adds a **goal region** where the target agents should be herded.
    The goal is represented:
    - In **Matplotlib**: As a semi-transparent green circle.
    - In **Pygame**: As a semi-transparent shaded circle.

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

    Notes
    -----
    - This renderer **modifies the post-render hooks** to include a goal region.
    - The **goal region is dynamically updated** at each frame based on `goal_pos` and `goal_radius`.
    - This class is designed specifically for **shepherding tasks** where agents must be guided to a target.

    Examples
    --------
    Example usage:

    .. code-block:: python

        from swarmsim.Renderers import ShepherdingRenderer

        renderer = ShepherdingRenderer(populations, environment, config_path="config.yaml")
        renderer.render()
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

        self.sensing_radius = self.config.get('sensing_radius', float('inf'))
        self.goal_circle = None

    def post_render_hook_matplotlib(self):
        """
        Adds the goal region rendering in Matplotlib.

        This function draws a **semi-transparent green circle** at the goal position,
        visually indicating where the target agents should be herded.

        Notes
        -----
        - Uses `plt.Circle` to draw the goal region.
        - The `alpha` value makes the goal **semi-transparent**.
        - The circle **automatically updates** if the goal moves.

        Examples
        --------
        The goal region is added **after** agents are plotted:

        .. code-block:: python

            goal_circle = plt.Circle(self.environment.goal_pos, self.environment.goal_radius,
                                     color='green', alpha=0.3, label='Goal Region')
            self.ax.add_artist(goal_circle)
        """
        if self.goal_circle is None:
            self.goal_circle = plt.Circle(self.environment.goal_pos,
                                          self.environment.goal_radius,
                                          color='green', alpha=0.3, label='Goal Region')
            self.ax.add_artist(self.goal_circle)
        else:
            self.goal_circle.center = self.environment.goal_pos
            self.goal_circle.set_radius(self.environment.goal_radius)

        # Draw herder angular sectors

        import numpy as np
        herder_positions = self.populations[1].x[:, :2]
        herder_vectors = herder_positions - self.environment.goal_pos
        herder_angles = np.arctan2(herder_vectors[:, 1], herder_vectors[:, 0]) % (2 * np.pi)
        NH = herder_positions.shape[0]

        # Clear previously drawn sector lines (orange-colored)
        lines_to_remove = [line for line in self.ax.lines if line.get_color() == 'orange']
        for line in lines_to_remove:
            line.remove()

        LB = np.zeros(NH)
        UB = np.zeros(NH)
        for j in range(NH):
            j_prev = (j - 1) % NH
            j_next = (j + 1) % NH
            zeta_minus = (herder_angles[j] - herder_angles[j_prev]) % (2 * np.pi)
            zeta_plus = (herder_angles[j_next] - herder_angles[j]) % (2 * np.pi)
            LB[j] = (herder_angles[j] - zeta_minus / 2) % (2 * np.pi)
            UB[j] = (herder_angles[j] + zeta_plus / 2) % (2 * np.pi)

            radius = 25

            # Plot boundary lines
            x1_lb, y1_lb = radius * np.cos(LB[j]), radius * np.sin(LB[j])
            x1_ub, y1_ub = radius * np.cos(UB[j]), radius * np.sin(UB[j])
            self.ax.plot([0, x1_lb], [0, y1_lb], color='orange', linestyle='--', linewidth=1)
            self.ax.plot([0, x1_ub], [0, y1_ub], color='orange', linestyle='--', linewidth=1)



    def post_render_hook_pygame(self):
        """
        Adds the goal region rendering in Pygame.

        This function draws a **semi-transparent shaded circle** at the goal position,
        visually indicating where the target agents should be herded.

        Notes
        -----
        - Uses `pygame.Surface` to create a **shaded overlay**.
        - The **goal region is semi-transparent**, allowing agents to remain visible.
        - The shading is dynamically adjusted based on the **arena size** and **goal radius**.

        Examples
        --------
        The goal region is added **after** agents are plotted:

        .. code-block:: python

            shading_surface = pygame.Surface((self.screen_size[0], self.screen_size[1]), pygame.SRCALPHA)
            shading_color = (0, 255, 0, 100)  # Green color with alpha = 100 (semi-transparent)
            pygame.draw.circle(shading_surface, shading_color, (goal_x, goal_y), goal_radius)
            self.window.blit(shading_surface, (0, 0))
        """


        # Convert goal position to screen coordinates
        goal_x = int((self.environment.goal_pos[0] + self.environment.dimensions[0] / 2) * self.scale_factor)
        goal_y = int((self.environment.dimensions[1] / 2 - self.environment.goal_pos[1]) * self.scale_factor)
        goal_radius = int(self.environment.goal_radius * self.scale_factor)

        # Create a semi-transparent surface
        shading_surface = pygame.Surface((self.screen_size[0], self.screen_size[1]), pygame.SRCALPHA)
        shading_color = (0, 255, 0, 100)  # Green color with alpha = 100 (semi-transparent)

        # Draw the goal region
        pygame.draw.circle(shading_surface, shading_color, (goal_x, goal_y), goal_radius)

        # Overlay the shading onto the main window
        self.window.blit(shading_surface, (0, 0))

        # -------- Draw the herders' sensing areas --------
        if self.sensing_radius < float('inf'):
            sensing_radius_sim = 10
            sensing_radius_screen = int(sensing_radius_sim * self.scale_factor)
            # Define the sensing overlay color (blue with low opacity)
            sensing_color = (0, 0, 255, 20)  # Blue with alpha = 50 (semi-transparent)

            # Retrieve herder positions
            herder_positions = self.populations[1].x[:, :2]
            for pos in herder_positions:
                # Convert simulation coordinates to screen coordinates
                x = int((pos[0] + self.environment.dimensions[0] / 2) * self.scale_factor)
                y = int((self.environment.dimensions[1] / 2 - pos[1]) * self.scale_factor)

                # Create a surface for the sensing area overlay
                sensing_surface = pygame.Surface((sensing_radius_screen * 2, sensing_radius_screen * 2), pygame.SRCALPHA)
                pygame.draw.circle(sensing_surface, sensing_color,
                                   (sensing_radius_screen, sensing_radius_screen),
                                   sensing_radius_screen)
                # Blit the sensing overlay so that it centers on the herder
                self.window.blit(sensing_surface, (x - sensing_radius_screen, y - sensing_radius_screen))
