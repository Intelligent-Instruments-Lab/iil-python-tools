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
knurl = { 'freq': 0.0, 'amp': 0.0, 'filter': 0.0, 'lin': 0.0 }

@swim
def gui_loop(d=0.5, i=0):
    world.process()
    world.window.show()
    a(gui_loop, d=1/16, i=i+1)

def knurl_handler(*args):
    world.boids.separate[None] = args[0] # 0.1 # freq
    world.boids.align[None]    = args[2] # 0.5 # amp
    world.boids.cohere[None]   = args[2] # 0.2 # filter
    world.boids.dt[None]       = args[3] * 4 # 0.2 # spec flat
    world.physarum.sense_angle[None] = args[0] * 0.30 * np.pi + 0.1
    world.physarum.sense_dist[None]  = args[2] * 20.0 + 1
    world.physarum.move_angle[None]  = args[2] * 0.3 * np.pi + 0.1
    world.physarum.move_step[None]   = args[3] * 5.0 + 1
    world.physarum.substep[None]     = int( args[3] * 20.0 + 1)
    print('knurl', args, world.physarum.substep[None])

# def knurl_handler(*args):
#     knurl['freq'] = args[0]
#     knurl['amp'] = args[1]
#     knurl['filter'] = args[2]
#     knurl['lin'] = args[3]
#     print('knurl', knurl)
#     knurl_map()

# def knurl_map():
#     world.boids.separate[None]      = knurl['freq'] # 0.1
#     world.boids.align[None]         = knurl['freq'] # 0.5
#     world.boids.cohere[None]        = knurl['freq'] # 0.2
#     world.boids.radius[None]        = knurl['freq'] * 100
#     world.boids.dt[None]            = knurl['freq'] * 3
#     world.boids.speed[None]         = knurl['freq'] * 5
#     world.physarum.sense_dist[None] = knurl['freq']*100 # 50.0
#     world.physarum.move_step[None]  = knurl['freq']*4 # 0.5
#     print('knurl mapped')

receive = Receiver(9998)
receive.attach('/', knurl_handler)


world.reset()

@swim
def param_loop(d=16, i=0):
    world.boids.radius[None] = P('50*cos($/3)')
    world.boids.dt[None]     = 0.5#P('2*cos($/9)')
    # world.boids.speed[None]  = P('2*sin($/7)')
    world.boids.separate[None] = 0.1#P('0.5+0.5*sin($/5)')
    world.boids.align[None]    = P('0.5+0.5*cos($/3)')
    world.boids.cohere[None]   = P('0.5+0.5*sin($/4)')
    world.physarum.sense_angle[None]    = P('1+0.2*sin($/3)*3.14') # 0.20 * np.pi
    world.physarum.sense_dist[None]     = P('2+sin($/4)') # 4.0
    world.physarum.evaporation[None]    = P('0.9, 0.999, 0.85, 0.95, 0.99') # 0.95
    world.physarum.move_angle[None]     = P('1+0.5*sin($/2)*3.14') # 0.1 * np.pi
    world.physarum.move_step[None]      = P('1,2,4,3,5,3',i) # 2.0
    world.physarum.substep[None] = 1#P('1,4,3',i) # 10
    a(param_loop, d=8, i=i+1)

hush(param_loop)
