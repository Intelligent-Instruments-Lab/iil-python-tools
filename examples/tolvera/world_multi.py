import taichi as ti
import numpy as np
import math
import tolvera as tol

from iipyper import OSC, run, repeat, cleanup
from iipyper.state import _lock

def main(host="127.0.0.1", port=7563):
    osc = OSC(host, port, verbose=False, concurrent=True)
    osc.create_client("world", host="127.0.0.1", port=7564)

    ti.init(arch=ti.vulkan)
    x = 1920
    y = 1080
    n = 4096
    world = tol.WorldMulti(x, y, n)
    # world_params = world.get_params()
    # boids_params = world.boids.get_params()
    # physarum_params = world.physarum.get_params()

    # @osc.args("/tolvera/boids/params")
    # def _(address, *args):
    #     nonlocal boids_params
    #     boids_params = args

    # @osc.args("/tolvera/physarum/params")
    # def _(address, *args):
    #     nonlocal physarum_params
    #     physarum_params = args

    # @osc.args("/tolvera/world/params")
    # def _(address, *args):
    #     nonlocal world_params
    #     world_params = args

    @osc.args("/tolvera/world/reset")
    def _(address, *args):
        world.reset()

    # count   = 5
    # counter = 0

    while world.window.running:
        with _lock:
            # if counter == count:
            #     counter = 0
            #     world.set_params(world_params)
            #     world.boids.set_params(boids_params)
            #     world.physarum.set_params(physarum_params)
            # counter +=1
            world.process()
            world.window.show()

if __name__ == '__main__':
    run(main)
