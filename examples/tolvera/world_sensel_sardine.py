from sardine import *
import taichi as ti
import tolvera as tol
import numpy as np
from pysensel import SenselOSC

def handle_contact(*args):
    sensel.handle_contact(args)

sensel = SenselOSC(7006, handle_contact)

ti.init(arch=ti.vulkan)
x=1920
y=1080
n=8192
world = tol.World(x, y, n)
c.bpm = 250
c.link()

@swim
def gui_loop(d=0.5, i=0):
    world.process()
    world.window.show()
    a(gui_loop, d=1/16, i=i+1)

@swim
def map_loop(d=16, i=0):
    world.boids.radius[None]         = sensel.contacts[0]['x']*20 # 10
    world.boids.dt[None]             = sensel.contacts[0]['y'] # 1
    world.boids.speed[None]          = (1-sensel.contacts[0]['f'])*2 # 2
    world.boids.separate[None]       = sensel.contacts[1]['x'] # 0.3
    world.boids.align[None]          = sensel.contacts[1]['y'] # 0.5
    world.boids.cohere[None]         = sensel.contacts[1]['f'] # 0.4
    world.physarum.sense_angle[None] = sensel.contacts[2]['x'] * np.pi # 0.1 * np.pi
    world.physarum.sense_dist[None]  = sensel.contacts[2]['y']*50 # 50.0
    # world.physarum.evaporation[None] = sensel.contacts[2]['f'] # 0.97
    # world.physarum.move_angle[None]  = sensel.contacts[1]['x'] # 0.1 * np.pi
    # world.physarum.move_step[None]   = sensel.contacts[1]['x'] # 0.5
    world.physarum.substep[None]     =1# sensel.contacts[1]['x'] # 2
    a(map_loop, d=1/16, i=i+1)

# hush(map_loop)

world.reset()
