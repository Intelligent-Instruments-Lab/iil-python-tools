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
    n = 8192
    boids = tol.vera.BoidsMulti(x, y, n, radius=10, colormode='rgb', species=2, size=1)
    # boids_params = boids.get_params()
    window = ti.ui.Window("Boids", (x, y))
    canvas = window.get_canvas()

    # @osc.args("/tolvera/boids/params")
    # def _(address, *args):
    #     nonlocal boids_params
    #     boids_params = args

    @osc.args("/tolvera/boids/reset")
    def _(address, *args):
        boids.reset()

    # count   = 5
    # counter = 0

    while window.running:
        with _lock:
            # if counter == count:
            #     counter = 0
            #     boids.set_params(boids_params)
            # counter +=1
            canvas.set_image(boids.process())
            window.show()

if __name__ == '__main__':
    run(main)
