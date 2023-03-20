import taichi as ti

from iipyper import OSC, run, repeat, cleanup
from iipyper.state import _lock

from tolvera.vera.particles import Particle, Particles
from tolvera.vera.pixels import Pixels
from tolvera.utils import Updater, UpdaterOSC

@ti.dataclass
class SimpleParticle:
    particle: Particle
    other_property: ti.f32

@ti.kernel
def randomise(p: ti.template(), n: int, x: int, y: int):
    for i in range(n):
        p.field[i] = SimpleParticle(
            particle=Particle(
                pos=[x*ti.random(ti.f32),y*ti.random(ti.f32)],
                vel=[2*(ti.random(ti.f32)-0.5), 2*(ti.random(ti.f32)-0.5)],
                active=1,
                size=5,
                rgba=[ti.random(ti.f32),ti.random(ti.f32),ti.random(ti.f32),1],
                speed=1))

def main(host="127.0.0.1", port=4000):
    ti.init(arch=ti.vulkan)
    osc = OSC(host, port, verbose=False, concurrent=True)
    x = 1920
    y = 1080
    n = 8
    p = Particles(SimpleParticle, x, y, n)
    randomise(p,n,x,y)
    window = ti.ui.Window("Particles", (x, y))
    canvas = window.get_canvas()
    px = Pixels(x,y)

    def update_pos(s):
        p.field[s[0]].particle.pos = [s[1], s[2]]
    def update_vel(s):
        p.field[s[0]].particle.vel = [s[1], s[2]]
    updaters = [
        UpdaterOSC(osc, "/tolvera/obstacles/pos", update_pos),
        UpdaterOSC(osc, "/tolvera/obstacles/vel", update_vel)
    ]

    while window.running:
        with _lock:
            # px.diffuse()
            # px.decay()
            [u() for u in updaters]
            px.clear()
            p(px())
            canvas.set_image(px())
            window.show()

if __name__ == '__main__':
    main()
