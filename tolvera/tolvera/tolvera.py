from tolvera.vera import *

import taichi as ti
import numpy as np
from tolvera.vera import *

"""
OSC input protocol

/tolvera/boids/radius i 0-500
/tolvera/boids/dt f 0-1
/tolvera/boids/speed f 0-10
/tolvera/boids/separate f 0-1
/tolvera/boids/align f 0-1
/tolvera/boids/cohere f 0-1
/tolvera/physarum/sense_angle f 0-2
/tolvera/physarum/sense_dist f 0-100
/tolvera/physarum/evaporation f 0.5-0.999
/tolvera/physarum/move_angle f 0-2 
/tolvera/physarum/move_step f 0-10
/tolvera/physarum/substep i 1-32
"""

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
        self.boids = Boids(x, y, n, species=5, colormode='g')
        self.physarum = Physarum(x, y, n)
        self.rea_diff = ReactionDiffusion(x, y)
        self.window = ti.ui.Window("tolvera", (x, y))#, fullscreen=True)
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
        self.physarum.deposit(self.boids._pos, 1.0)
        # self.physarum.stamp_pxg(self.rea_diff.px_g, 0.1, 1.0)
        # self.physarum.deposit_swarm(self.boids._pos, self.boids._vel, 4.0)

    def process(self):
        self.interact()
        self.boids.process()
        # self.rea_diff.process()
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
    n = 16384
    world = World(x, y, n)
    world.reset()
    world.boids.radius[None] = 10
    world.boids.dt[None] = 1
    world.boids.speed[None] = 2
    world.boids.separate[None] = 0.3
    world.boids.align[None] = 0.5
    world.boids.cohere[None] = 0.4
    world.physarum.sense_angle[None] = 0.1 * np.pi
    world.physarum.sense_dist[None] = 50.0
    world.physarum.evaporation[None] = 0.97
    world.physarum.move_angle[None] = 0.1 * np.pi
    world.physarum.move_step[None] = 0.5
    world.physarum.substep[None] = 2
    while world.window.running:
        world.process()
        world.window.show()

if __name__ == '__main__':
    main()
