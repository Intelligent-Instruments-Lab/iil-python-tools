import taichi as ti
import numpy as np
import math
import tolvera as tol

from iipyper import OSC, run, repeat, cleanup
from iipyper.state import _lock

def main(host="127.0.0.1", port=4000):
    osc = OSC(host, port, verbose=False, concurrent=True)
    osc.create_client("boids", host="127.0.0.1", port=7564)
    
    ti.init(arch=ti.vulkan)    
    x = 512
    y = 512
    n = 256
    s = 2
    boids = tol.vera.BoidsMulti(x, y, n, radius=10, colormode='rgb', species=s, size=1)
    # alive = 0
    # for b in range(n):
    #     if (b % s) == 0:
    #         boids._alive[0, b] = 0.0
    #         if (b % 16) == 0:
    #             boids._alive[0, b] = 1.0
    #             alive +=1
    # boids._alive[0, 0] = 1.0
    # print('alive',alive)
    for b in range(n):
        boids._alive[0, b] = b % s
    boids._alive[0, 0] = 1.0
    
    boids_set_params_species = 0
    boids_set_params = boids.get_params(boids_set_params_species)
    window = ti.ui.Window("Boids", (x, y))
    canvas = window.get_canvas()

    @osc.args("/tolvera/boids/params")
    def _(address, *args):
        nonlocal boids_set_params, boids_set_params_species
        boids_set_params_species = args[0]
        boids_set_params = args[1:]
        print(boids_set_params_species, boids_set_params)

    @osc.args("/tolvera/boids/reset")
    def _(address, *args):
        boids.reset()

    @osc.args("/tolvera/boids/set")
    def _(address, *args):
        # args: [id, pos_x, pos_y, vel_x, vel_y]
        boids.set_boid(args[0], args[1:])

    @osc.args("/tolvera/boids/get")
    def _(address, *args):
        return boids.get_boid(args[0])

    count   = 5
    counter = 0

    while window.running:
        with _lock:
            if counter == count:
                counter = 0
                boids.set_params(boids_set_params_species, boids_set_params)
                # osc("boids", "/tolvera/boids/get", boids.get_boid(0))
                # print(boids.get_boid(0))
            counter +=1
            canvas.set_image(boids.process())
            window.show()

if __name__ == '__main__':
    run(main)
