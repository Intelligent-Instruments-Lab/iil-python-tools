import taichi as ti
import numpy as np
import math
import tolvera as tol

from iipyper import OSC, run, repeat, cleanup
from iipyper.state import _lock

def main(host="127.0.0.1", port=7563):
    osc = OSC(host, port, verbose=False, concurrent=True)
    osc.create_client("boids", host="127.0.0.1", port=7564)
    
    ti.init(arch=ti.vulkan)    
    x = 1920
    y = 1080
    n = 4096
    boids = tol.vera.Boids(x, y, n, radius=10, colormode='rgb', species=5)
    boids_params = boids.get_params()
    window = ti.ui.Window("Boids", (x, y))
    canvas = window.get_canvas()

    @osc.args("/tolvera/boids/params")
    def _(address, *args):
        nonlocal boids_params
        boids_params = args

    @osc.args("/tolvera/boids/reset")
    def _(address, *args):
        boids.init()

    @osc.args("/tolvera/boids/set_boid")
    def _(address, *args):
        # args: [id, pos_x, pos_y, vel_x, vel_y]
        boids.set_boid(args[0], args[1:])

    @osc.args("/tolvera/boids/boid/get")
    def _(address, *args):
        return boids.get_boid(args[0])

    count   = 5
    counter = 0

    while window.running:
        with _lock:
            if counter == count:
                counter = 0
                boids.set_params(boids_params)
                # osc("boids", "/tolvera/boids/get", boids.get_boid(0))
                # print(boids.get_boid(0))
            counter +=1
            canvas.set_image(boids.process())
            window.show()

if __name__ == '__main__':
    run(main)
