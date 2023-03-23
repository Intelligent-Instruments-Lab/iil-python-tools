import numpy as np
import time
import taichi as ti

from iipyper import OSC, run, repeat, cleanup
from iipyper.state import _lock

from tolvera.particles import Particles
from tolvera.pixels import Pixels
from tolvera.utils import OSCUpdaters

def main(host="127.0.0.1", port=4000):
    ti.init(arch=ti.vulkan)
    np.random.seed(int(time.time()))
    # ti.init()
    osc = OSC(host, port, verbose=False, concurrent=True)
    osc.create_client("particles", host="127.0.0.1", port=7564)
    x = int(1920)
    y = int(1080)
    n = 1024
    p = Particles(x,y,n)
    window = ti.ui.Window("Particles", (x, y), fps_limit=120)
    canvas = window.get_canvas()
    px = Pixels(x,y, evaporate=0.9)

    # o = OSCUpdaters(osc, client="particles",
    #     receives={
    #         "/tolvera/particles/set/pos": p.osc_set_pos,
    #         "/tolvera/particles/set/vel": p.osc_set_vel
    #     }, receive_count=10,
    #     sends={
    #         "/tolvera/particles/get/pos/all": p.osc_get_pos_all
    #     }, send_count=60
    # )

    while window.running:
        with _lock:
            px.diffuse()
            # px.decay()
            # px.clear()
            # o()
            p(px())
            canvas.set_image(px())
            window.show()

if __name__ == '__main__':
    main()
