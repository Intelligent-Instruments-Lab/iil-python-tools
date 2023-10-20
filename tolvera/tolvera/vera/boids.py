'''
TODO: Custom wall behaviour
'''

import time
import taichi as ti

from tolvera.utils import init, render
from tolvera.particles import Particles
from tolvera.pixels import Pixels
from tolvera.consts import BOIDS as BOIDS_CONSTS
from iipyper import OSC, run, Updater

@ti.dataclass
class BoidsRules:
    separate: ti.f32
    align:    ti.f32
    cohere:   ti.f32
    radius:   ti.f32

@ti.data_oriented
class Boids():
    def __init__(self,
                 x: ti.i32,
                 y: ti.i32,
                 species: ti.i32 = 3) -> None:
        self.x = x
        self.y = y
        self.species_n = species
        self.rules = BoidsRules.field(shape=(species,species))
        self.consts = BOIDS_CONSTS()
        self.init()
    @ti.kernel
    def init(self):
        for i in range(self.species_n):
            self.init_rules(i)
    @ti.func
    def init_rules(self, i: ti.i32):
        for j in range(self.species_n):
            self.rules[i,j] = BoidsRules(
                separate= ti.random(ti.f32),
                align   = ti.random(ti.f32), 
                cohere  = ti.random(ti.f32),
                radius  = ti.random(ti.f32)*self.consts.RADIUS_MAX)
    # TODO: for i, j in range(field.shape[0], field.shape[0]):
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
        r = BoidsRules()
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
        c = self.consts
        self.rules[i,j] = BoidsRules(separate, align, cohere, radius*c.RADIUS_MAX)
    @ti.kernel
    def set_all_rules(self, separate: ti.template(), align: ti.template(), cohere: ti.template(), radii: ti.template()):
        for i in range(self.species_n):
            self._set_all_rules(i, separate, align, cohere, radii)
    @ti.func
    def _set_all_rules(self, i, separate, align, cohere, radii):
        c = self.consts
        for j in range(self.species_n):
            self.rules[i,j] = BoidsRules(separate[i,j], align[i,j], cohere[i,j], radii[i,j]*c.RADIUS_MAX)
    @ti.kernel
    def set_all_rules_from_list(self, rules: ti.types.ndarray()):
        assert rules.shape[0] == self.species_n * self.species_n * 4, f"rules.shape[0]={rules.shape[0]} != {self.species_n * self.species_n * 4}"
        for i in range(self.species_n):
            self._set_all_rules_from_list(i, rules)
    @ti.func
    def _set_all_rules_from_list(self, i, rules: ti.types.ndarray()):
        c = self.consts
        for j in range(self.species_n):
            self.rules[i,j] = BoidsRules(rules[i*4+j+0], rules[i*4+j+1], rules[i*4+j+2], rules[i*4+j+3]*c.RADIUS_MAX)
    @ti.kernel
    def set_species_rules(self, i: ti.i32, separate: ti.template(), align: ti.template(), cohere: ti.template(), radii: ti.template()):
        c = self.consts
        for j in range(self.species_n):
            self.rules[i,j] = BoidsRules(separate[j], align[j], cohere[j], radii[j]*c.RADIUS_MAX)
    @ti.kernel
    def set_species_rules_from_list(self, i: ti.i32, rules: ti.types.ndarray()):
        c = self.consts
        for j in range(self.species_n):
            self.rules[i,j] = BoidsRules(rules[j*4+0], rules[j*4+1], rules[j*4+2], rules[j*4+3]*c.RADIUS_MAX)
    def get_rule(self, i, j):
        return self.rules[i,j].to_numpy().tolist()

def main(**kwargs):
    o = init(**kwargs)
    particles = Particles(o.x, o.y, o.n, o.species)
    pixels = Pixels(o.x, o.y, evaporate=o.evaporate, fps=o.fps)
    boids = Boids(o.x, o.y, o.species)

    def reset():
        particles.reset()
        pixels.reset()
        boids.reset()
    update = Updater(reset, o.fps*8)

    def _():
        update()
        pixels.diffuse()
        boids(particles)
        particles(pixels)

    render(_, pixels)

if __name__ == '__main__':
    run(main)
