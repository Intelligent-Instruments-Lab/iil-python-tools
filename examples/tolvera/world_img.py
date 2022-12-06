import taichi as ti
import tolvera as tol
import numpy as np

# TODO: Abstract out image reader + add functions?

def main():
    ti.init(arch=ti.vulkan)
    x = 1920
    y = 1080
    n = 8192
    # path = 'examples/tolvera/img/kandinsky.jpeg'
    # path = 'examples/tolvera/img/cardew.jpeg'
    path = 'examples/tolvera/img/lines.png'
    img = ti.tools.imread(path)[:,:,0]
    img_fld = ti.field(dtype=ti.f32, shape=img.shape)
    img_fld.from_numpy(img)
    world = tol.World(x, y, n)
    # world.boids.radius[None] = 10
    # world.boids.dt[None] = 1
    # world.boids.speed[None] = 2
    # world.boids.separate[None] = 0.3
    # world.boids.align[None] = 0.5
    # world.boids.cohere[None] = 0.4
    world.physarum.sense_angle[None] = 0.2 * np.pi
    world.physarum.sense_dist[None] = 10.0
    world.physarum.evaporation[None] = 0.95
    world.physarum.move_angle[None] = 0.2 * np.pi
    world.physarum.move_step[None] = 2
    world.physarum.substep[None] = 1
    while world.window.running:
        # world.physarum.stamp(img_fld, 0.8, 0.2) # kandinsky
        world.physarum.stamp(img_fld, 0.8, 3) # lines
        # world.physarum.stamp(img_fld, 0.8, 1) # cardew
        world.process()
        world.window.show()

if __name__ == '__main__':
    main()
