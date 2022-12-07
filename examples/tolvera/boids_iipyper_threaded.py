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
    # n = 16785408 # 8392704 # 4196352 # 2098176 # 1049088 # 524544 # 262272 # 131136 # 
    # n = 65568 # 
    n = 32784 # 
    # n = 16384
    boids = tol.vera.Boids(x, y, n, radius=10, colormode='rgb', species=10, size=1)
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

    count   = 5
    counter = 0

    while window.running:
        with _lock:
            if counter == count:
                counter = 0
                boids.update(params)
            counter +=1
            canvas.set_image(boids.process())
            window.show()

if __name__ == '__main__':
    main()
