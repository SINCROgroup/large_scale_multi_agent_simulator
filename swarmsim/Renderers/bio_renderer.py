import matplotlib.pyplot as plt
import numpy as np
import pygame
from swarmsim.Renderers import BaseRenderer



class BioRenderer(BaseRenderer):

    def __init__(self, populations, environment=None, config_path=None):
        super().__init__(populations, environment, config_path)
        self.screen_size = (1366,768)
        self.arena_size = (1366,768)

    

    def render_pygame(self):
              
        super().render_pygame()

    


