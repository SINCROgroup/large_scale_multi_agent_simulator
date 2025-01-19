import matplotlib.pyplot as plt
import pygame
from swarmsim.Renderers import BaseRenderer


class ShepherdingRenderer(BaseRenderer):
    """
    A renderer that extends BaseRenderer to include rendering of a goal region for shepherding tasks.
    """

    def post_render_hook_matplotlib(self):
        """Add the goal region rendering in Matplotlib."""
        goal_circle = plt.Circle(self.environment.goal_pos, self.environment.goal_radius,
                                 color='green', alpha=0.3, label='Goal Region')
        self.ax.add_artist(goal_circle)

    def post_render_hook_pygame(self):
        """Add the goal region rendering in Pygame with shading."""
        scale = min(self.arena_size[0] / self.environment.dimensions[0],
                    self.arena_size[1] / self.environment.dimensions[1])
        goal_x = int((self.environment.goal_pos[0] + self.environment.dimensions[0] / 2) * scale) + 100
        goal_y = int((self.environment.dimensions[1] / 2 - self.environment.goal_pos[1]) * scale) + 100
        goal_radius = int(self.environment.goal_radius * scale)

        # Create a surface with per-pixel alpha for shading
        shading_surface = pygame.Surface((self.screen_size[0], self.screen_size[1]), pygame.SRCALPHA)
        shading_color = (0, 255, 0, 100)  # Green color with alpha = 100 (semi-transparent)

        # Draw a shaded circle on the surface
        pygame.draw.circle(shading_surface, shading_color, (goal_x, goal_y), goal_radius)

        # Blit the shaded surface onto the main window
        self.window.blit(shading_surface, (0, 0))
