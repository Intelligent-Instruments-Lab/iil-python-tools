import taichi as ti
import numpy as np
import math
import tolvera as tol
from mrp import MRP

from iipyper import OSC, run, repeat, cleanup
from iipyper.state import _lock

def main(host="127.0.0.1", receive_port=7563, send_port=7770):
    osc = OSC(host=host, port=receive_port, verbose=False, concurrent=True)
    osc.create_client("mrp", host, send_port)

    ti.init(arch=ti.vulkan)
    x = 1920
    y = 1080
    n = 88
    world = tol.World(x, y, n)
    world_params = world.get_params()
    boids_params = world.boids.get_params()
    physarum_params = world.physarum.get_params()
    
    count   = 5
    counter = 0
    
    mrp = None

    @osc.args("/tolvera/boids/params")
    def _(address, *args):
        nonlocal boids_params
        boids_params = args

    @osc.args("/tolvera/physarum/params")
    def _(address, *args):
        nonlocal physarum_params
        physarum_params = args

    @osc.args("/tolvera/world/params")
    def _(address, *args):
        nonlocal world_params
        world_params = args

    @osc.args("/reset")
    def reset(address, kind=None):
        nonlocal mrp
        world.reset()
        mrp = MRP(osc)
        mrp.all_notes_off()

    @cleanup
    def _():
        mrp.cleanup()

    reset(None)

    while world.window.running:
        with _lock:
            if counter == count:
                counter = 0
                world.set_params(world_params)
                world.boids.set_params(boids_params)
                world.physarum.set_params(physarum_params)
            counter +=1
            world.process()
            world.window.show()

if __name__ == '__main__':
    run(main)
