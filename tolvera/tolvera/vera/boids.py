import taichi as ti

from tolvera.vera.particles import Particle, Particles
from tolvera.vera.pixels import Pixels

from iipyper import OSC, run, repeat, cleanup
from iipyper.state import _lock

vec1 = ti.types.vector(1, ti.f32)
vec2 = ti.math.vec2
vec3 = ti.math.vec3
vec4 = ti.math.vec4

@ti.dataclass
class BoidsParticle:
    particle: Particle
    align:    ti.f32
    separate: ti.f32
    cohere:   ti.f32
    radius:   ti.f32

@ti.data_oriented
class Boids(Particles):
    def __init__(self,
                 max_n: ti.i32,
                 x: ti.i32,
                 y: ti.i32) -> None:
        super().__init__(BoidsParticle, x, y, max_n)
    @ti.kernel
    def randomise(self, n: int):
        for i in range(n):
            p = Particle(
                    pos=[self.x*ti.random(ti.f32),self.y*ti.random(ti.f32)],
                    active=1,size=2,speed=1,max_speed=2)
            p.vel=[p.max_speed * (ti.random(ti.f32)-0.5),p.max_speed * (ti.random(ti.f32)-0.5)]
            p.rgba=[1,1,1,1]
            # p.rgba=[ti.random(ti.f32),
            #         ti.random(ti.f32),
            #         ti.random(ti.f32),1]
            self.field[i] = BoidsParticle(
                separate=0.2,#ti.random(ti.f32),
                align=0.7,#ti.random(ti.f32),
                cohere=0.7,#,ti.random(ti.f32),
                radius=50,#ti.random(ti.f32)*100,
                particle=p)
    @ti.kernel
    def step(self):
        for i in range(self.max_n):
            if self.field[i].particle.active > 0.0:
                self.step_inner(i)
    @ti.func
    def step_inner(self, i: ti.i32):
        # FIXME: repro disappearing dataclass @ti.func 'bp.dist'
        b = self.field[i]
        bp = b.particle
        separate = ti.Vector([0.,0.])
        align    = ti.Vector([0.,0.])
        cohere   = ti.Vector([0.,0.])
        nearby = 0
        for i_near in range(self.max_n):
            if i!=i_near and self.field[i_near].particle.active > 0.0:
                dis = self.field[i].particle.dist(
                    self.field[i_near].particle)
                dis_norm = dis.norm()
                if dis_norm < b.radius:
                    self.field[i].particle.vel += \
                        dis.normalized()/dis_norm * bp.max_speed
                    separate += dis
                    align    += bp.vel
                    cohere   += bp.pos
                    nearby   += 1
        if nearby != 0:
            separate = separate/nearby        * b.separate
            align    = align/nearby           * b.align
            cohere   = (cohere/nearby-bp.pos) * b.cohere
            self.field[i].particle.vel += \
                (cohere + align + separate).normalized()
            self.limit_speed(i)

def main(host="127.0.0.1", port=4000):
    ti.init(arch=ti.vulkan)
    osc = OSC(host, port, verbose=False, concurrent=True)
    x = 1920
    y = 1080
    n = 1024
    b = Boids(n, x, y)
    b.randomise(n)
    px = Pixels(x,y)
    window = ti.ui.Window("Boids", (x, y))
    canvas = window.get_canvas()

    while window.running:
        with _lock:
            # px.diffuse()
            # px.decay()
            px.clear()
            b(px())
            canvas.set_image(px())
            window.show()

if __name__ == '__main__':
    main()