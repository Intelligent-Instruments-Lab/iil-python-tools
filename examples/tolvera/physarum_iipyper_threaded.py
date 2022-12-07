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
    n = 16785408 # 8392704 # 4196352 # 2098176 # 1049088 # 524544 # 262272 # 131136 # 65568 # 32784
    physarum = tol.vera.Physarum(x, y, n)
    window = ti.ui.Window("Physarum", (x, y))
    canvas = window.get_canvas()

    # TODO: Should this be a struct for different types?
    params = ti.field(dtype=ti.f32, shape=(6))
    # TODO: Replace with a getter
    params[0] = physarum.sense_angle[None]
    params[1] = physarum.sense_dist[None]
    params[2] = physarum.evaporation[None]
    params[3] = physarum.move_angle[None]
    params[4] = physarum.move_step[None]
    params[5] = physarum.substep[None]
    print(params)

    @osc.args("/physarum/params")
    def _(address, *args):
        nonlocal params
        params = args

    count   = 5
    counter = 0

    while window.running:
        with _lock:
            if counter == count:
                counter = 0
                physarum.update(params)
            counter +=1
            canvas.set_image(physarum.process())
            window.show()

if __name__ == '__main__':
    main()
