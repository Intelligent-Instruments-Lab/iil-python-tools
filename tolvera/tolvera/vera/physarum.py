'''
TODO: Custom wall behaviour
TODO: Colour sensing -> species sensing
TODO: Rendering brightness (?) 
'''

import taichi as ti

from tolvera.utils import *
from tolvera.particles import Particles
from tolvera.pixels import Pixels
from iipyper import run, Updater

@ti.dataclass
class PhysarumParams:
    sense_angle: ti.f32
    sense_dist:  ti.f32
    move_angle:  ti.f32
    move_dist:   ti.f32

@ti.data_oriented
class Physarum():
    def __init__(self,
                 trail_x: ti.i32,
                 trail_y: ti.i32,
                 species: ti.i32 = 3,
                 evaporate: ti.f32 = 0.95,
                 substep: ti.i32 = 1) -> None:
        self.species_n = species
        self.rules = PhysarumParams.field(shape=(species))
        self.rule_consts = {
            'sense_angle': 0.3 * ti.math.pi,
            'move_angle':  0.3 * ti.math.pi,
            'sense_dist':  50.0,
            'move_dist':   1.0,
            'dist_min':    0.1,
        }
        self.x = trail_x
        self.y = trail_y
        self.substep = substep
        self.evaporate = ti.field(ti.f32, shape=())
        self.evaporate[None] = evaporate
        self.trail = Pixels(self.x, self.y, evaporate=evaporate)
        self.init()
    @ti.kernel
    def init(self):
        self.init_rules()
    @ti.func
    def init_rules(self):
        c = self.rule_consts
        for i in range(self.species_n):
            self.rules[i] = PhysarumParams(
                sense_angle = ti.random(ti.f32) * c['sense_angle'],
                move_angle  = ti.random(ti.f32) * c['move_angle'],
                sense_dist  = ti.random(ti.f32) * c['sense_dist'] + c['dist_min'],
                move_dist   = ti.random(ti.f32) * c['move_dist'] + c['dist_min'])
    @ti.kernel
    def move(self, field: ti.template()):
        for i in range(field.shape[0]):
            if field[i].active > 0.0:
                pos, ang = field[i].pos, field[i].ang
                rule = self.rules[field[i].species]
                c = self.sense(pos, ang,                    rule.sense_dist).norm()
                l = self.sense(pos, ang - rule.sense_angle, rule.sense_dist).norm()
                r = self.sense(pos, ang + rule.sense_angle, rule.sense_dist).norm()
                if l < c < r:
                    ang += rule.move_angle
                elif l > c > r:
                    ang -= rule.move_angle
                elif r > c and c < l:
                    ang += rule.move_angle * (2 * (ti.random() < 0.5) - 1) # TODO: magic numbers
                pos += ti.Vector([ti.cos(ang), ti.sin(ang)]) \
                    * rule.move_dist * field[i].active
                field[i].left = l
                field[i].centre = c
                field[i].right = r
                field[i].pos = pos
                field[i].ang = ang
    @ti.func
    def sense(self, pos, ang, dist):
        # TODO: speciate based on rgba compared with field[i].rgba
        p = pos + ti.Vector([ti.cos(ang), ti.sin(ang)]) * dist
        px = ti.cast(p[0], ti.i32) % self.x
        py = ti.cast(p[1], ti.i32) % self.y
        return self.trail.px.rgba[px, py]
    @ti.kernel
    def deposit(self, field: ti.template()):
        for i in range(field.shape[0]):
            if field[i].active > 0.0:
                p = field[i]
                x = ti.cast(p.pos[0], ti.i32) % self.x
                y = ti.cast(p.pos[1], ti.i32) % self.y
                self.trail.circle(x, y, p.size, p.rgba * p.active)
    @ti.kernel
    def deposit_px(self, px: ti.template(), weight: ti.f32):
        for i, j in ti.ndrange(self.x, self.y):
            p = px[0,i,j]
            if p > 0.0:
                self.trail.px.rgba[i, j] += ti.Vector([p,p,p,1]) * weight
    def step(self, field):
        self.deposit(field)
        self.trail.diffuse()
    def process(self, particles):
        for i in range(self.substep):
            self.move(particles.field)
            self.step(particles.field)
    def reset(self):
        self.init()
    def set_evaporate(self, evaporate):
        self.evaporate[None] = evaporate
        self.trail.evaporate[None] = evaporate
    @ti.kernel
    def set_rule(self, i: ti.i32, sense_angle: ti.f32, sense_dist: ti.f32, move_angle: ti.f32, move_dist: ti.f32):
        c = self.rule_consts
        self.rules[i] = PhysarumParams(
            sense_angle * c['sense_angle'],
            sense_dist  * c['sense_dist'] + c['dist_min'],
            move_angle  * c['move_angle'],
            move_dist   * c['move_dist'] + c['dist_min'])
    @ti.kernel
    def set_all_rules(self, sense_angle: ti.template(), sense_dist: ti.template(), move_angle: ti.template(), move_dist: ti.template()):
        c = self.rule_consts
        for i in range(self.species_n):
            self.rules[i] = PhysarumParams(
                sense_angle[i] * c['sense_angle'],
                sense_dist[i]  * c['sense_dist'] + c['dist_min'],
                move_angle[i]  * c['move_angle'],
                move_dist[i]   * c['move_dist'] + c['dist_min'])
    @ti.kernel
    def set_all_rules_from_list(self, rules: ti.types.ndarray()):
        c = self.rule_consts
        for i in range(self.species_n):
            self.rules[i] = PhysarumParams(
                rules[i*4+0] * c['sense_angle'],
                rules[i*4+1] * c['sense_dist'] + c['dist_min'],
                rules[i*4+2] * c['move_angle'],
                rules[i*4+3] * c['move_dist'] + c['dist_min'])
    def get_rule(self, i, j):
        return self.rules[i,j].to_numpy().tolist()
    def __call__(self, particles):
        self.process(particles)
        return self.trail.px.rgba

def main():
    init()
    particles = Particles(x, y, n, species)
    pixels = Pixels(x,y)
    physarum = Physarum(x, y, species)

    def reset():
        particles.reset()
        pixels.reset()
        physarum.reset()
    update = Updater(reset, fps*4)

    def _():
        update()
        physarum(particles)
        particles.activity_decay()
        pixels.set(physarum.trail.px)

    render(_, pixels)

if __name__ == '__main__':
    run(main)
