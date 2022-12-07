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
    boids = tol.vera.Boids(x, y, n, colormode='rgb', species=30)
    window = ti.ui.Window("Boids", (x, y))
    canvas = window.get_canvas()

    params = ti.field(dtype=ti.f32, shape=(7))
    params[0] = boids.separate[None]
    params[1] = boids.align[None]
    params[2] = boids.cohere[None]
    params[3] = boids.fear[None]
    params[4] = boids.dt[None]
    params[5] = boids.radius[None]
    params[6] = boids.speed[None]

    @osc.args("/boids/params")
    def _(address, *args):
        nonlocal params
        params = args

    @ti.kernel
    def update(p: ti.template()):
        boids.separate[None] = p[0]
        boids.align[None]    = p[1]
        boids.cohere[None]   = p[2]
        boids.fear[None]     = p[3]
        boids.dt[None]       = p[4]
        boids.radius[None]   = p[5]
        boids.speed[None]    = p[6]

    count   = 5
    counter = 0

    while window.running:
        with _lock:
            if counter == count:
                counter = 0
                update(params)
            counter +=1
            canvas.set_image(boids.process())
            window.show()

if __name__ == '__main__':
    main()
