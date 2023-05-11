import taichi as ti
from tolvera.particles import Particle

# TODO: render()
# TODO: more neighbour funcs/stats
# TODO: move particles.seek|avoid here? (attract|repel)
# TODO: rename to Target?

@ti.dataclass
class Attractor:
    p: Particle
    radius: ti.f32

@ti.data_oriented
class Attractors:
    def __init__(self, x=1920, y=1080, n=1) -> None:
        self.x = x
        self.y = y
        self.n = n
        self.field = Attractor.field(shape=(n))
    def set(self, i, attractor: Attractor):
        self.field[i] = attractor
    def get(self, i):
        return self.field[i]
    @ti.kernel
    def randomise(self):
        for i in range(self.n):
            self.field[i].p.pos = ti.Vector([ti.random() * self.x, ti.random() * self.y])
            self.field[i].p.mass = ti.random() * 1.0
            self.field[i].radius = ti.random() * self.y
    @ti.kernel
    def nn(self, field: ti.template()):
        for i in range(self.n):
            if self.field[i].p.active > 0.0:
                self.nn_inner(field, i)
    @ti.func
    def nn_inner(self, field: ti.template(), i: ti.i32):
        a = self.field[i]
        nearby = 0
        for j in range(field.shape[0]):
            p = field[j]
            if p.active > 0.0:
                dis = self.field[i].p.dist(p).norm()
                if dis < a.radius:
                    nearby += 1
        if nearby != 0:
            self.field[i].p.nearby = nearby
    # @ti.kernel
    # def render(self, pixels):
    #     # draw circle with no fill
    #     # color based on p.mass
    #     # fill based on p.nearby
    #     # radius based on radius
    #     pass
    def __call__(self, particles):
        self.nn(particles.field)

