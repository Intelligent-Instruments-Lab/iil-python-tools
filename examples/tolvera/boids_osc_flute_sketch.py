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
    x = 1920
    y = 1080
    n = 2048
    boids = tol.vera.BoidsMulti(x, y, n, radius=10, colormode='rgb', species=1, size=1)
    boids_params = boids.get_params(0)
    window = ti.ui.Window("Boids", (x, y))
    canvas = window.get_canvas()

    flute_data = boids.get_params(0)
    print(flute_data)

    @osc.args("/tolvera/boids/params")
    def _(address, *args):
        nonlocal boids_params
        boids_params = args

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

    @osc.args("/flute")
    def _(address, *args):
        nonlocal flute_data
        flute_data = args

    count   = 5
    counter = 0

    while window.running:
        with _lock:
            if counter == count:
                counter = 0
                boids.set_params(0, flute_data)
                # osc("boids", "/tolvera/boids/get", boids.get_boid(0))
                # print(boids.get_boid(0))
            counter +=1
            canvas.set_image(boids.process())
            window.show()

if __name__ == '__main__':
    run(main)
