import time
import taichi as ti

from iipyper import OSC, run

from tolvera.particles import Particles
from tolvera.pixels import Pixels
from tolvera.utils import OSCUpdaters

def main(x=1920, y=1080, n=1024, species=4, fps=120, host="127.0.0.1", port=4000):
    seed = int(time.time())
    ti.init(arch=ti.vulkan, random_seed=seed)
    # ti.init(random_seed=seed)
    osc = OSC(host, port, verbose=False, concurrent=True)
    osc.create_client("particles", host="127.0.0.1", port=7564)
    particles = Particles(x,y,n,species)
    pixels = Pixels(x,y, evaporate=0.9, fps=fps)

    # particles.activate(5, (x/2, y/2), species=0)
    # particles.deactivate(5, species=0)

    osc_update = OSCUpdaters(osc, client="particles",
        receives={
            "/tolvera/particles/set/pos": particles.osc_set_pos,
            "/tolvera/particles/set/vel": particles.osc_set_vel
        }, receive_count=10,
        sends={
            "/tolvera/particles/get/pos/all": particles.osc_get_pos_all
        }, send_count=60
    )

    '''
    particles(n=1024, species=5)
        .flock()
        .slime()
    '''

    def render():
        # pixels.diffuse()
        # pixels.decay()
        pixels.clear()
        # osc_update()
        particles(pixels)

    pixels.show(render)

if __name__ == '__main__':
    run(main())
