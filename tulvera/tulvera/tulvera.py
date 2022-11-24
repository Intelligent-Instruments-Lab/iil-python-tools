from tulvera.vera import *

# TODO: Use Profiler to understand why this cuts framerate by half:

import taichi as ti
import numpy as np
from tulvera.vera import *

@ti.data_oriented
class World:
    def __init__(self,
                 x=1024,
                 y=1024,
                 n=1024):
        self.boids = Boids(x, y, n)
        self.physarum = Physarum(x, y, n)
        self.window = ti.ui.Window("Tulvera", (x, y))
        self.canvas = self.window.get_canvas()
        self.init()

    @ti.kernel
    def init(self):
        self.pause(False)

    def process(self):
        self.boids.process()
        self.physarum.process()
        self.canvas.set_image(self.boids.world.to_numpy()[0])

    def pause(self, val):
        self.boids.pause = val
        self.physarum.pause = val
    
    def cleanup():
        pass

def main():
    ti.init(arch=ti.vulkan)
    x = 1920
    y = 1080
    n = 4096
    world = World(x, y, n)
    # window = ti.ui.Window("World", (x, y))
    # canvas = window.get_canvas()
    while world.window.running:
        world.process()
        # update(boids) # jurigged: costs 10fps
        # world.canvas.set_image(world.boids.world.to_numpy()[0])
        world.window.show()

if __name__ == '__main__':
    main()
