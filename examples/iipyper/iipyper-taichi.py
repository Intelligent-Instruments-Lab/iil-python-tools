import random

import taichi as ti
import taichi.math as tm

from iipyper import OSC, repeat, run, cleanup, Lag

def main(osc_host='127.0.0.1', osc_port=9999, size=512, verbose=True):

    osc = OSC(osc_host, osc_port, verbose=verbose)
    osc.create_client('self', port=osc_port) # uses same host as server

    ti.init(arch=ti.gpu)

    pixels = ti.Vector.field(3, dtype=float, shape=(size * 2, size))

    @ti.func
    def complex_sqr(z):  # complex square of a 2D vector
        return tm.vec2(z[0] * z[0] - z[1] * z[1], 2 * z[0] * z[1])

    @ti.kernel
    def paint(t: float, zoom: float):
        for i, j in pixels:  # Parallelized over all pixels
            c = tm.vec2(-0.8, tm.cos(t) * 0.2)
            z = tm.vec2(i / size - 1, j / size - 0.5) * 2 / zoom
            iterations = 0
            while z.norm() < 50 and iterations < 50:
                z = complex_sqr(z) + c
                iterations += 1
            pixels[i, j] = 1 - tm.vec3(
                iterations * 0.02, z.norm() * 0.02, iterations * 0.015)

    @cleanup
    def _():
        print('exiting...')

    @repeat(1/100)
    def _():
        osc('self', '/control', 
            random.random()*2-1,
            2**((random.random()-0.5)*0.1)
            )

    idx = Lag(.9, val=0)
    zoom = Lag(.7, val=1)

    @osc.args
    def control(addr, i, z):
        idx(idx.val + i)
        zoom(zoom.val * z)

    gui = ti.GUI("Julia Set", res=(size * 2, size), fast_gui=True)
    while gui.running:
        paint(idx.val, zoom.val)
        gui.set_image(pixels)
        gui.show()


if __name__=='__main__':
    run(main)