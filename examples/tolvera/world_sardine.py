from sardine import *
import taichi as ti
import tolvera as tol
import numpy as np

ti.init(arch=ti.vulkan)
x=1920
y=540
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

disc = {
    'ang': {'x':0.0,'y':0.0,'z':0.0},
    'acc': {'x':0.0,'y':0.0,'z':0.0}
}

def magnetic_handler(*args):
    print('/magnetic:')
    disc['ang'] = {'x':args[0],'y':args[1],'z':args[2]}
    disc['acc'] = {'x':args[3],'y':args[4],'z':args[5]}
    print('disc', disc['ang'], disc['acc'])
    disc_map()

def disc_map():
    world.boids.radius[None] = disc['ang']['x'] * 100
    world.boids.dt[None] = disc['ang']['y'] * 3
    world.boids.speed[None] = disc['ang']['z'] * 5
    world.boids.separate[None] = disc['ang']['x'] # 0.1
    world.boids.align[None]    = disc['ang']['y'] # 0.5
    world.boids.cohere[None]   = disc['ang']['z'] # 0.2
    world.physarum.sense_dist[None] = disc['acc']['x']*100 # 50.0
    world.physarum.move_step[None]  = disc['acc']['y']*4 # 0.5
    world.physarum.substep[None]    = 1+int(disc['acc']['z']*8) # 2
    print('mapped')

receive = Receiver(port=7563, ip="192.168.0.100", name="amagnetic")
receive = Receiver(7400)
receive.attach('/magnetic', magnetic_handler)

c.bpm = 250
c.link()

@swim
def gui_loop(d=0.5, i=0):
    world.process()
    world.window.show()
    a(gui_loop, d=1/16, i=i+1)

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
    # world.physarum.move_angle[None]     = P('1+0.5*sin($/2)*3.14') # 0.1 * np.pi
    world.physarum.move_step[None]      = P('1,2,4,3,5,3',i) # 2.0
    world.physarum.substep[None] = 1#P('1,4,3',i) # 10
    a(param_loop, d=8, i=i+1)

hush(param_loop)
