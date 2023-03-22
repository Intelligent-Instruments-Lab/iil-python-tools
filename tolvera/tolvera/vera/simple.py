import taichi as ti

from iipyper import OSC, run, repeat, cleanup
from iipyper.state import _lock

from tolvera.vera.particles import Particle, Particles
from tolvera.vera.pixels import Pixels
from tolvera.utils import OSCUpdaters

@ti.dataclass
class SimpleParticle:
    particle: Particle
    other_property: ti.f32

@ti.data_oriented
class SimpleParticles(Particles):
    def __init__(self, x: ti.i32, y: ti.i32, max_n: ti.i32):
        super().__init__(SimpleParticle, x, y, max_n)
    @ti.kernel
    def randomise(self, n: int):
        for i in range(n):
            self.field[i] = SimpleParticle(
                particle=Particle(
                    pos=[self.x*ti.random(ti.f32),self.y*ti.random(ti.f32)],
                    vel=[2*(ti.random(ti.f32)-0.5), 2*(ti.random(ti.f32)-0.5)],
                    rgba=[ti.random(ti.f32),ti.random(ti.f32),ti.random(ti.f32),1],
                    active=1, size=2, speed=1),
                other_property=ti.random(ti.f32))

def main(host="127.0.0.1", port=4000):
    ti.init(arch=ti.vulkan)
    osc = OSC(host, port, verbose=False, concurrent=True)
    osc.create_client("particles", host="127.0.0.1", port=7564)
    x = 1920
    y = 1080
    n = 8
    s = SimpleParticles(x, y, n)
    s.randomise(n)
    window = ti.ui.Window("SimpleParticles", (x, y))
    canvas = window.get_canvas()
    px = Pixels(x,y)

    o = OSCUpdaters(osc, client="particles",
        receives={
            "/tolvera/particles/set/pos": s.osc_set_pos,
            "/tolvera/particles/set/vel": s.osc_set_vel
        }, receive_count=10,
        sends={
            "/tolvera/particles/get/pos/all": s.osc_get_pos_all
        }, send_count=60
    )

    while window.running:
        with _lock:
            px.diffuse()
            # px.decay()
            # px.clear()
            o()
            s(px())
            canvas.set_image(px())
            window.show()

if __name__ == '__main__':
    main()
