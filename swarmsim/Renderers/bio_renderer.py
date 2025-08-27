import numpy as np
import pygame
from swarmsim.Renderers import BaseRenderer

from swarmsim.Environments import Environment
from swarmsim.Controllers import Controller



class BioRenderer(BaseRenderer):
    """
    Specialized renderer for bio-inspired simulations with spatial field visualization.

    This renderer extends the base renderer to visualize spatial control fields and
    environmental gradients that influence bio-inspired agent behavior. It creates
    real-time spatial intensity maps showing how controllers respond to different
    spatial locations, providing insight into bio-inspired navigation and decision-making.

    The renderer generates a continuous spatial field visualization by:
    
    1. **Grid Sampling**: Creates a dense grid across the environment
    2. **Controller Evaluation**: Queries the controller at each grid point
    3. **Color Mapping**: Maps controller responses to visual intensity
    4. **Real-time Update**: Updates the spatial field each frame

    Parameters
    ----------
    populations : list of Population
        List of agent populations to visualize.
    environment : Environment
        The simulation environment instance.
    config_path : str
        Path to YAML configuration file with rendering parameters.
    controller : Controller
        Controller instance used to generate spatial field visualization.

    Attributes
    ----------
    controller : Controller
        The controller providing spatial field information.
    screen_size : tuple
        Pygame window dimensions (width, height). Default: (1366, 768).
    arena_size : tuple  
        Arena rendering dimensions. Default: (1366, 768).
    spatial_input : pygame.Surface
        Cached surface containing the spatial field visualization.

    Implementation Details
    ----------------------
    The spatial field is computed by:
    
    1. Creating a uniform grid spanning the environment dimensions
    2. Evaluating ``controller.get_action_in_space()`` at each grid point
    3. Mapping controller responses to RGB color values
    4. Generating a pygame surface for efficient blitting

    Color Mapping
    -------------
    - **High Response**: Red color (255, 0, 0) for maximum controller activation
    - **Low Response**: White color (255, 255, 255) for minimum activation  
    - **Interpolation**: Linear blending between red and white based on response magnitude

    Examples
    --------
    Basic bio-renderer setup:

    .. code-block:: python

        from swarmsim.Renderers import BioRenderer
        from swarmsim.Controllers import LightSensitiveController

        controller = LightSensitiveController(config_path="bio_config.yaml")
        renderer = BioRenderer(populations, environment, "render_config.yaml", controller)
        
        # Render with spatial field background
        renderer.render()

    Configuration example:

    .. code-block:: yaml

        BioRenderer:
            render_mode: "pygame"
            render_dt: 0.05
            agent_colors: ["blue", "green"]
            agent_sizes: [2, 3]
            background_color: "black"

    Notes
    -----
    - Requires pygame rendering mode for spatial field visualization
    - Controller must implement ``get_action_in_space(positions)`` method
    - Spatial field is recomputed each frame if controller state changes
    - Best suited for visualizing spatially-varying control policies
    """

    def __init__(self, populations: list, environment: Environment, config_path: str, controller: Controller):
        """
        Initialize the bio-renderer with spatial field visualization capabilities.

        Parameters
        ----------
        populations : list of Population
            List of agent populations to visualize.
        environment : Environment
            The simulation environment instance.
        config_path : str
            Path to YAML configuration file containing rendering parameters.
        controller : Controller
            Controller instance for generating spatial field visualization.
            Must implement ``get_action_in_space(positions)`` method.

        Notes
        -----
        - Initializes spatial field surface during construction
        - Controller is used immediately to generate initial spatial visualization
        """
        super().__init__(populations, environment, config_path)
        self.controller = controller
        self.screen_size = (1366,768)
        self.arena_size = (1366,768)
        self.spatial_input = None
        self.create_spatial_input()


    

    def pre_render_hook_pygame(self):
        """
        Render the spatial field background before drawing agents.

        This method applies the cached spatial field surface as a background,
        providing visual context for bio-inspired agent behavior. The spatial
        field shows the controller's response across the entire environment.

        Implementation
        --------------
        - Blits the pre-computed spatial field surface to the pygame display
        - Called automatically before agent rendering in each frame
        - Uses cached surface for optimal performance

        Notes
        -----
        - Spatial field surface is generated by ``create_spatial_input()``
        - Background visualization updates when controller state changes
        - Provides continuous spatial context for discrete agent positions
        """

        # Draw surface
        pygame.display.get_surface().blit(self.spatial_input, (0, 0))

    def create_spatial_input(self):
        """
        Generate spatial field visualization surface from controller responses.

        This method creates a high-resolution spatial field by evaluating the
        controller at every pixel location and mapping the responses to colors.
        The resulting surface provides a continuous visualization of how the
        controller responds across the entire environment space.

        Algorithm
        ---------
        1. **Grid Generation**: Create uniform grid spanning environment dimensions
        2. **Coordinate Mapping**: Map screen pixels to simulation coordinates  
        3. **Controller Evaluation**: Query controller response at each position
        4. **Color Interpolation**: Blend red (high) and white (low) based on response
        5. **Surface Creation**: Generate pygame surface for efficient rendering

        Mathematical Details
        --------------------
        Coordinate transformation:
        
        .. math::
            x_{sim} = x_{inf} + \\frac{i}{w} (x_{sup} - x_{inf})
            
            y_{sim} = y_{inf} + \\frac{j}{h} (y_{sup} - y_{inf})

        Where :math:`(i,j)` are pixel coordinates and :math:`(w,h)` are screen dimensions.

        Color blending:
        
        .. math::
            RGB(i,j) = \\text{intensity}(i,j) \\cdot [255,0,0] + (1-\\text{intensity}(i,j)) \\cdot [255,255,255]

        
        Notes
        -----
        - Called during initialization and when controller state changes
        - Handles controllers returning None by creating default visualization
        - Uses pygame's surface array interface for efficient pixel manipulation
        - Assumes controller output is normalized to [0,1] range
        """

        x_inf = - self.environment.dimensions[0] / 2
        x_sup = self.environment.dimensions[0] / 2
        y_inf = - self.environment.dimensions[1] / 2
        y_sup = self.environment.dimensions[1] / 2


        # Create a grid of points
        x = np.linspace(x_inf, x_sup, self.screen_size[0])  
        y = np.linspace(y_inf, y_sup, self.screen_size[1])  

        # Create the 2D grid of values
        X, Y = np.meshgrid(x, y)  # Create a grid from x and y

        Pixels_positions = np.column_stack((X.T.ravel(), Y.T.ravel()))
        
        if self.controller is None:
            # If no controller is provided, create a default spatial input
            Pixel_ity = np.zeros((self.screen_size[0] * self.screen_size[1], 1))
        else:
            Pixel_ity = self.controller.get_action_in_space(Pixels_positions)
        Pixel_ity = Pixel_ity.reshape(self.screen_size[0],self.screen_size[1])


        # Create a new 3D matrix to store the blended colors (RGB channels)
        RGB_image = np.zeros((self.screen_size[0],self.screen_size[1], 3))

        color1 = np.array([255,0,0])
        color2 = np.array([255,255,255])

        # Blend each pixel
        for i in range(self.screen_size[0]):
            for j in range(self.screen_size[1]):
                RGB_image[i, j , :] = (Pixel_ity[i, j] * color1 + (1 - Pixel_ity[i, j]) * color2)



        # Create a Pygame surface
        self.spatial_input = pygame.surfarray.make_surface(RGB_image)








    


