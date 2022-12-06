import taichi as ti
import numpy as np
import math

from tolvera.vera._particle import Particles

"""

"""

@ti.data_oriented
class Obstacles(Particles):
    def __init__(self,
                 x=1024,
                 y=1024,
                 n=8,
                 r=32):
        super().__init__(x, y, n)
        # make rects/cirlces
        # velocity?
        # are obs just other boids?
        self.randomise()
    
    @ti.kernel
    def randomise(self):
        # for o in range(self._pos):
        pass

