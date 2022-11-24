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
        # TODO: n for each Particle vera?
        self._x = x
        self._y = y
        self._n = n
        self.boids = Boids(x, y, n)
        self.physarum = Physarum(x, y, n)
        self.rea_diff = ReactionDiffusion(x, y)
        self.window = ti.ui.Window("Tulvera", (x, y))
        self.canvas = self.window.get_canvas()
        self.init()

    @ti.kernel
    def init(self):
        self.pause(False)

    def pause(self, val):
        self.boids.pause = val
        self.physarum.pause = val

    def speed(self, val):
        self.boids.dt[None] = val
        self.physarum.move_step[None] = ti.cast(val*10, ti.i32)
        # self.physarum.substep[None] = ti.cast(val*10, ti.i32)

    def interact(self):
        self.physarum.deposit(self.boids._pos)

    def process(self):
        self.interact()
        self.boids.process()
        self.rea_diff.process()
        self.canvas.set_image(1-self.physarum.process())

    def reset(self):
        self.boids.init()
        self.physarum.init()
    
    def cleanup():
        pass

def main():
    ti.init(arch=ti.vulkan)
    x = 1920
    y = 1080
    n = 4096
    world = World(x, y, n)
    world.reset()
    while world.window.running:
        world.process()
        world.window.show()

if __name__ == '__main__':
    main()
