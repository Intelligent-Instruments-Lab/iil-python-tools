import taichi as ti
import numpy as np
import time

from iipyper import OSC, run, repeat, cleanup
from iipyper.state import _lock

# TODO: abstract out ti.ui.Window? fps_limit
# TODO: color palette
# FIXME: @ti.dataclass inheritance https://github.com/taichi-dev/taichi/issues/7422

vec1 = ti.types.vector(1, ti.f32)
vec2 = ti.math.vec2
vec3 = ti.math.vec3
vec4 = ti.math.vec4

# TODO: Decouple Species from Particle?
# @ti.dataclass
# class Species:
#     size:      ti.f32
#     speed:     ti.f32
#     max_speed: ti.f32
#     mass:      ti.f32
#     rgba:      vec4
#     diffusion: ti.f32 # for pixels

@ti.dataclass
class Particle:
    # g:    vec1
    # rgb:  vec3
    rgba: vec4
    pos:  vec2
    vel:  vec2
    # acc:  vec2
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
                 x: ti.i32, y: ti.i32,
                 max_n: ti.i32,
                 species: ti.i32 = 1,
                 substep: ti.i32 = 1):
        self.max_n = max_n
        self.species_n = species
        self.species_c = ti.Vector.field(4, ti.f32, shape=(species))
        self.x = x
        self.y = y
        self.field = Particle.field(shape=(self.max_n))
        self.substep = substep
        np.random.seed(int(time.time()))
        self.init()
    @ti.kernel
    def init(self):
        self.randomise()
        self.speciate()
    @ti.func
    def randomise(self):
        for i in range(self.max_n):
            self.field[i] = Particle(
                pos=[self.x*ti.random(ti.f32),self.y*ti.random(ti.f32)],
                vel=[2*(ti.random(ti.f32)-0.5), 2*(ti.random(ti.f32)-0.5)],
                active=1)
    @ti.func
    def speciate(self):
        for i in range(self.species_n):
            self.species_c[i] = [ti.random(ti.f32),ti.random(ti.f32),ti.random(ti.f32),1]
        for i in range(self.max_n):
            s = i % self.species_n
            self.field[i].species = s
            self.field[i].rgba    = self.species_c[s]
            self.field[i].size      = 1 * ti.random(ti.f32) + 1.0
            self.field[i].speed     = 1 * ti.random(ti.f32) + 0.2
            self.field[i].max_speed = 2 * ti.random(ti.f32) + 1.0
            self.field[i].mass      = 1 * ti.random(ti.f32) + 1.0
    @ti.kernel
    def move(self):
        for i in range(self.max_n):
            p = self.field[i]
            if p.active > 0.0:
                self.field[i].pos += p.speed * p.vel
                x = p.pos[0]
                y = p.pos[1]
                if  (x > self.x): self.field.pos[i][0] = 1
                elif(y > self.y): self.field.pos[i][1] = 1
                elif(x < 0):      self.field.pos[i][0] = self.x-1
                elif(y < 0):      self.field.pos[i][1] = self.y-1
                # TODO: wall repel
                # w = 10
                # if  (x > self.x-w): self.field.vel[i][0] *= -1
                # elif(y > self.y-w): self.field.vel[i][1] *= -1
                # elif(x < w):        self.field.vel[i][0] *= -1
                # elif(y < w):        self.field.vel[i][1] *= -1
                self.limit_speed(i)
    @ti.func
    def limit_speed(self, i: int):
        p = self.field[i]
        if p.vel.norm() > p.max_speed:
            self.field[i].vel = p.vel.normalized() * p.max_speed
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
            p = self.field[i]
            if p.active > 0.0:
                xi = ti.cast(p.pos[0], ti.i32) - p.size
                xj = ti.cast(p.pos[0], ti.i32) + p.size
                yi = ti.cast(p.pos[1], ti.i32) - p.size
                yj = ti.cast(p.pos[1], ti.i32) + p.size
                for x in range(xi, xj):
                    for y in range(yi, yj):
                        px[x, y] = p.rgba
    def osc_set_pos(self, i, x, y):
        self.field[i].pos = [x, y]
    def osc_set_vel(self, i, x, y):
        self.field[i].vel = [x, y]
    def osc_get_pos(self, i):
        return self.field[i].pos.to_numpy().tolist()
    def osc_get_vel(self, i):
        return self.field[i].vel.to_numpy().tolist()
    @ti.kernel
    def osc_get_pos_all(self):
        # TODO: parellelise for loop
        pos = ti.field(ti.f32, shape=self.max_n*2)
        j = 0
        for i in range(self.max_n):
            p = self.field[i]
            if p.active > 0.0:
                pos[j] = p.pos[0]
                j+=1
                pos[j] = p.pos[1]
                j+=1
        return pos.to_numpy().tolist()
    def reset(self):
        self.init()
    def __call__(self, px):
        self.process()
        self.render(px)
