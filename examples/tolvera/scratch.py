from sardine import *
import taichi as ti
import tolvera as tol
import numpy as np

from senselosc import Sensel as sensel

def mapping(world, sensel):
    world.boids.radius[None]   = sensel # 10
    world.boids.dt[None]       = sensel # 1
    world.boids.speed[None]    = sensel # 2
    world.boids.separate[None] = sensel # 0.3
    world.boids.align[None]    = sensel # 0.5
    world.boids.cohere[None]   = sensel # 0.4
    world.physarum.sense_angle[None] = sensel # 0.1 * np.pi
    world.physarum.sense_dist[None]  = sensel # 50.0
    world.physarum.evaporation[None] = sensel # 0.97
    world.physarum.move_angle[None]  = sensel # 0.1 * np.pi
    world.physarum.move_step[None]   = sensel # 0.5
    world.physarum.substep[None]     = sensel # 2
    pass

def main():
    ti.init(arch=ti.vulkan)
    x = 1920
    y = 1080
    n = 16384
    world = tol.World(x, y, n)
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

# world.save||load(i)
# world.reset()
# world.clear()

# world.pause(True)
# world.speed(s)

# world.spawn||kill('Boids||Physarum||All', x, y, r, q)
# world.draw||erase('Noise||Attractor||Repeller||Membrane||Food', x, y, r)
# world.decay||metabolise(0.5)

# world.import||mask('image')

# world.sample()
# world.trace||track()
# 

# world.interact(['Boids', 'Physarum'], rule="separate", weight=0.2)

# world.
# world.sample()

