import time
import taichi as ti

from tolvera.utils import *
from tolvera.particles import Particles
from tolvera.pixels import Pixels

from iipyper import OSC, run, Updater

@ti.dataclass
class BoidsParams:
    separate: ti.f32
    align:    ti.f32
    cohere:   ti.f32
    radius:   ti.f32

@ti.data_oriented
class Boids():
    def __init__(self,
                 x: ti.i32,
                 y: ti.i32,
                 species=3) -> None:
        self.x = x
        self.y = y
        self.species_n = species
        self.rules = BoidsParams.field(shape=(species,species))
        self.radius_max = 300.0
        self.init()
    @ti.kernel
    def init(self):
        self.init_rules()
    @ti.func
    def init_rules(self):
        for i in range(self.species_n):
            for j in range(self.species_n):
                self.rules[i,j] = BoidsParams(
                    separate= ti.random(ti.f32),
                    align   = ti.random(ti.f32), 
                    cohere  = ti.random(ti.f32),
                    radius  = ti.random(ti.f32)*self.radius_max)
    @ti.kernel
    def step(self, field: ti.template()):
        for i in range(field.shape[0]):
            if field[i].active > 0.0:
                self.step_inner(field, i)
    @ti.func
    def step_inner(self, field: ti.template(), i: ti.i32):
        p1 = field[i]
        separate = ti.Vector([0.,0.])
        align    = ti.Vector([0.,0.])
        cohere   = ti.Vector([0.,0.])
        nearby = 0
        r = BoidsParams()
        for j in range(field.shape[0]):
            p2 = field[j]
            if i!=j and p2.active > 0.0:
                dis = field[i].dist(p2)
                dis_norm = dis.norm()
                r = self.rules[p1.species, p2.species]
                if dis_norm < r.radius:
                    field[i].vel += \
                        dis.normalized()/dis_norm * p1.max_speed
                    separate += dis
                    align    += p2.vel
                    cohere   += p2.pos
                    nearby   += 1
        if nearby != 0:
            separate = separate/nearby         * r.separate * p1.active
            align    = align/nearby            * r.align    * p1.active
            cohere   = (cohere/nearby-p1.pos)  * r.cohere   * p1.active
            field[i].nearby = nearby
            field[i].vel += (cohere+align+separate).normalized()
    def reset(self):
        self.init()
    def __call__(self, particles):
        self.step(particles.field)
    def set_rule(self, i, j, separate, align, cohere, radius):
        self.rules[i,j] = BoidsParams(separate, align, cohere, radius)
    @ti.kernel
    def set_all_rules(self, separate: ti.template(), align: ti.template(), cohere: ti.template(), radii: ti.template()):
        for i in range(self.species_n):
            self._set_all_rules(i, separate, align, cohere, radii)
    @ti.func
    def _set_all_rules(self, i, separate, align, cohere, radii):
        for j in range(self.species_n):
            self.rules[i,j] = BoidsParams(separate[i,j], align[i,j], cohere[i,j], radii[i,j])
    @ti.kernel
    def set_all_rules_from_list(self, rules: ti.types.ndarray()):
        assert rules.shape[0] == self.species_n * self.species_n * 4, f"rules.shape[0]={rules.shape[0]} != {self.species_n * self.species_n * 4}"
        for i in range(self.species_n):
            self._set_all_rules_from_list(i, rules)
    @ti.func
    def _set_all_rules_from_list(self, i, rules: ti.types.ndarray()):
        for j in range(self.species_n):
            self.rules[i,j] = BoidsParams(rules[i*4+j+0], rules[i*4+j+1], rules[i*4+j+2], rules[i*4+j+3]*self.radius_max)
    @ti.kernel
    def set_species_rules(self, i: ti.i32, separate: ti.template(), align: ti.template(), cohere: ti.template(), radii: ti.template()):
        for j in range(self.species_n):
            self.rules[i,j] = BoidsParams(separate[j], align[j], cohere[j], radii[j])
    @ti.kernel
    def set_species_rules_from_list(self, i: ti.i32, rules: ti.types.ndarray()):
        for j in range(self.species_n):
            self.rules[i,j] = BoidsParams(rules[j*4+0], rules[j*4+1], rules[j*4+2], rules[j*4+3]*self.radius_max)
    def get_rule(self, i, j):
        return self.rules[i,j].to_numpy().tolist()

def main():
    init()
    particles = Particles(x, y, n, species)
    pixels = Pixels(x, y, evaporate=0.95, fps=fps)
    boids = Boids(x, y, species)

    def reset():
        particles.reset()
        pixels.reset()
        boids.reset()
    update = Updater(reset, fps*8)

    def _():
        update()
        pixels.diffuse()
        boids(particles)
        particles(pixels)

    render(_, pixels)

if __name__ == '__main__':
    run(main)
