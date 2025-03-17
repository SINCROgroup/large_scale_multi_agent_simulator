import matplotlib.pyplot as plt
import numpy as np
import pygame
from swarmsim.Renderers import BaseRenderer



class BioRenderer(BaseRenderer):

    def __init__(self, populations, environment=None, config_path=None, controller=None):
        super().__init__(populations, environment, config_path)
        self.controller = controller
        self.screen_size = (1366,768)
        self.arena_size = (1366,768)
        self.create_spatial_input()

    

    def pre_render_hook_pygame(self):

        # Draw surface
        pygame.display.get_surface().blit(self.spatial_input, (0, 0))

    
    def create_spatial_input(self):

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








    


