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
    # n = 4196352 # 2098176 #
    n = 1049088 # 524544 # 262272 # 131136 # 65568 # 32784
    physarum = tol.vera.PhysarumMulti(x, y, n)
    # physarum_params = physarum.get_params()
    window = ti.ui.Window("Physarum", (x, y))
    canvas = window.get_canvas()

    # @osc.args("/tolvera/physarum/params")
    # def _(address, *args):
    #     nonlocal physarum_params
    #     physarum_params = args

    @osc.args("/tolvera/physarum/reset")
    def _(address, *args):
        physarum.reset()

    # count   = 5
    # counter = 0

    while window.running:
        with _lock:
            # if counter == count:
            #     counter = 0
            #     physarum.set_params(physarum_params)
            # counter +=1
            canvas.set_image(physarum.process())
            window.show()

if __name__ == '__main__':
    run(main)
