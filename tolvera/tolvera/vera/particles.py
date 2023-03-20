import taichi as ti

from iipyper import OSC, run, repeat, cleanup
from iipyper.state import _lock

# FIXME: @ti.dataclass inheritance https://github.com/taichi-dev/taichi/issues/7422

vec1 = ti.types.vector(1, ti.f32)
vec2 = ti.math.vec2
vec3 = ti.math.vec3
vec4 = ti.math.vec4

@ti.dataclass
class Particle:
    g:    vec1
    rgb:  vec3
    rgba: vec4
    pos:  vec2
    vel:  vec2
    acc:  vec2
    mass:      ti.f32
    size:      ti.f32
    speed:     ti.f32
    max_speed: ti.f32
    active:    ti.f32
    species:   ti.i32
    @ti.func
    def dist(self, other):
        # FIXME: wrap around walls
        return self.pos - other.pos
    @ti.func
    def dist_norm(self, other):
        return self.dist(self.pos - other.pos).norm()
    @ti.func
    def dist_normalized(self, other):
        return self.dist(self.pos - other.pos).normalized()

@ti.data_oriented
class Particles:
    def __init__(self, 
                 Type: ti.template(), 
                 x: ti.i32, y: ti.i32,
                 max_n: ti.i32,
                 substep: ti.i32 = 1):
        self.max_n = max_n
        self.x = x
        self.y = y
        # TODO: assert self.field[0].pos exists else warn use Particle
        self.type = Type
        self.field = Type.field(shape=(self.max_n))
        self.substep = substep
        self.init()
    def init(self):
        # self.randomise()
        pass
    @ti.kernel
    def randomise(self, n: int, x: int, y: int):
        for i in range(n):
            self.field[i] = self.type(
                particle=Particle(
                    pos=[x*ti.random(ti.f32),y*ti.random(ti.f32)],
                    vel=[2*(ti.random(ti.f32)-0.5), 2*(ti.random(ti.f32)-0.5)],
                    active=1,
                    size=1,
                    rgba=[ti.random(ti.f32),ti.random(ti.f32),ti.random(ti.f32),1],
                    speed=1))
    @ti.kernel
    def move(self):
        for i in range(self.max_n):
            p = self.field[i].particle
            if p.active > 0.0:
                self.field[i].particle.pos += p.speed * p.vel
                x = p.pos[0]
                y = p.pos[1]
                if  (x > self.x): self.field.particle.pos[i][0] = 1
                elif(y > self.y): self.field.particle.pos[i][1] = 1
                elif(x < 0):      self.field.particle.pos[i][0] = self.x-1
                elif(y < 0):      self.field.particle.pos[i][1] = self.y-1
    @ti.func
    def limit_speed(self, i: int):
        p = self.field[i].particle
        if p.vel.norm() > p.max_speed:
            self.field[i].particle.vel = p.vel.normalized() * p.max_speed
    @ti.kernel
    def step(self):
        pass
    def process(self):
        for i in range(self.substep):
            self.step()
            self.move()
    @ti.kernel
    def render(self, px: ti.template()):
        # TODO: support g, rgb, rgba
        # TODO: render shapes
        # l = len(px[0,0])
        for i in range(self.max_n):
            p = self.field[i].particle
            if p.active > 0.0:
                xi = ti.cast(p.pos[0], ti.i32) - p.size
                xj = ti.cast(p.pos[0], ti.i32) + p.size
                yi = ti.cast(p.pos[1], ti.i32) - p.size
                yj = ti.cast(p.pos[1], ti.i32) + p.size
                for x in range(xi, xj):
                    for y in range(yi, yj):
                        px[x, y] = p.rgba
    def __call__(self, px):
        self.process()
        self.render(px)

