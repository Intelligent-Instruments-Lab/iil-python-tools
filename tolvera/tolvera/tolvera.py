from tolvera.vera import *

import taichi as ti
import numpy as np
from tolvera.vera import *

"""
OSC input protocol

/tolvera/boids/params fffffff separate 0-1, align 0-1, cohere 0-1, fear 0-100, radius 0-500, dt 0-1, speed 0-10
/tolvera/physarum/params fffffi sense_angle 0-2, sense_dist 0-100, evaporation 0.5-0.999, move_angle 0-2 , move_step 0-10, substep 1-32
/tolvera/world/params f threshold, weight, etc
/tolvera/image/name s filename
/tolvera/image/params ff weight, threshold

OSC output protocol

/tolvera/boids/pos i
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
        self.physarum = Physarum(x, y, n*16)
        # self.rea_diff = ReactionDiffusion(x, y)
        self.window = ti.ui.Window("tolvera", (x, y))#, fullscreen=True)
        self.canvas = self.window.get_canvas()
        self.params = ti.field(ti.f32, shape=(6))
        # p0 = physarum.deposit(boids, weight)
        # p1 = 
        # p2 = 
        # p3 = 
        # p4 = 
        # p5 = 
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
        self.physarum.deposit(self.boids._pos, self.params[0])#2.0)
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

    @ti.kernel
    def set_params(self, p: ti.template()):
        # TODO: self.params = p # why does this assignment error?
        self.params[0] = p[0]
        self.params[1] = p[1]
        self.params[2] = p[2]
        self.params[3] = p[3]
        self.params[4] = p[4]
        self.params[5] = p[5]
    
    def get_params(self):
        return self.params

@ti.data_oriented
class WorldMulti:
    def __init__(self,
                 x=1024,
                 y=1024,
                 n=1024):
        # TODO: n for each Particle vera?
        self._x = x
        self._y = y
        self._n = n
        self.boids = BoidsMulti(x, y, n, species=4, colormode='rgb')
        self.physarum = PhysarumMulti(x, y, n*16, colormode='rgb', species=4)
        # self.rea_diff = ReactionDiffusion(x, y)
        self.window = ti.ui.Window("tolvera", (x, y))#, fullscreen=True)
        self.canvas = self.window.get_canvas()
        self.params = ti.field(ti.f32, shape=(6))
        # p0 = physarum.deposit(boids, weight)
        # p1 = 
        # p2 = 
        # p3 = 
        # p4 = 
        # p5 = 
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
        self.physarum.deposit(self.boids._pos, 8.0)#self.params[0])#2.0)
        # self.physarum.stamp_pxg(self.rea_diff.px_g, 0.1, 1.0)
        # self.physarum.deposit_swarm(self.boids._pos, self.boids._vel, 4.0)

    def process(self):
        self.interact()
        self.boids.process()
        # self.rea_diff.process()
        self.canvas.set_image(self.physarum.process())

    def reset(self):
        self.boids.reset()
        self.physarum.reset()
    
    def cleanup():
        pass

    @ti.kernel
    def set_params(self, p: ti.template()):
        # TODO: self.params = p # why does this assignment error?
        self.params[0] = p[0]
        self.params[1] = p[1]
        self.params[2] = p[2]
        self.params[3] = p[3]
        self.params[4] = p[4]
        self.params[5] = p[5]
    
    def get_params(self):
        return self.params


# def main():
#     ti.init(arch=ti.vulkan)
#     x = 1920
#     y = 1080
#     n = 8196
#     world = World(x, y, n)
#     world.reset()
#     world.boids.radius[None] = 30
#     world.boids.dt[None] = 1
#     world.boids.speed[None] = 2
#     world.boids.separate[None] = 0.1
#     world.boids.align[None] = 0.7
#     world.boids.cohere[None] = 0.5
#     world.physarum.sense_angle[None] = 0.2 * np.pi
#     world.physarum.sense_dist[None] = 40.0
#     world.physarum.evaporation[None] = 0.7
#     world.physarum.move_angle[None] = 0.2 * np.pi
#     world.physarum.move_step[None] = 1
#     world.physarum.substep[None] = 2
#     while world.window.running:
#         world.process()
#         world.window.show()

def main():
    ti.init(arch=ti.vulkan)
    x = 1920
    y = 1080
    n = 4096
    world = WorldMulti(x, y, n)
    # world.reset()
    while world.window.running:
        world.process()
        world.window.show()

if __name__ == '__main__':
    main()
