from sardine import *
import taichi as ti
import tolvera as tol
import numpy as np

ti.init(arch=ti.vulkan)
x=1920
y=1080
n=8192
world = tol.World(x, y, n)
world.boids.radius[None] = 10
world.boids.dt[None] = 1
world.boids.speed[None] = 2
world.boids.separate[None] = 0.1
world.boids.align[None] = 0.5
world.boids.cohere[None] = 0.2
world.physarum.sense_angle[None] = 0.1 * np.pi
world.physarum.sense_dist[None] = 50.0
world.physarum.evaporation[None] = 0.97
world.physarum.move_angle[None] = 0.1 * np.pi
world.physarum.move_step[None] = 0.5
world.physarum.substep[None] = 2
c.bpm = 250
c.link()

@swim
def gui_loop(d=0.5, i=0):
    world.process()
    world.window.show()
    a(gui_loop, d=1/16, i=i+1)

def muse_handler(*args):
    world.boids.radius[None]         = args[0] # boids_radius
    # world.boids.dt[None]             = args[1] # boids_dt
    world.boids.speed[None]          = args[2] # boids_speed
    world.boids.cohere[None]         = args[3] # boids_cohere
    world.boids.align[None]          = args[4] # boids_align
    world.boids.separate[None]       = args[5] # boids_separate
    world.physarum.sense_angle[None] = args[6] # phys_sense_angle
    world.physarum.sense_dist[None]  = args[7] # phys_sense_dist
    # world.physarum.evaporation[None] = args[8] # phys_evaporation
    world.physarum.move_angle[None]  = args[9] # phys_move_angle
    world.physarum.move_step[None]   = args[10] # phys_move_step
    # world.physarum.substep[None]     = args[11] # phys_substep
    # print('muse mapped', args)

receive = Receiver(7400)
receive.attach('/tolvera', muse_handler)

world.reset()
