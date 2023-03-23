import time
import numpy as np
import taichi as ti

from tolvera.particles import Particles
from tolvera.pixels import Pixels
from tolvera.utils import Updater, OSCUpdaters

from iipyper import OSC, run

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
                 max_n:   ti.i32,
                 species: ti.i32 = 3,
                 diffuse_amount: ti.f32 = 0.95,
                 substep: ti.i32 = 1) -> None:
        self.max_n = max_n
        self.species_n = species
        self.rules = PhysarumParams.field(shape=(self.max_n))
        self.heading = ti.field(dtype=ti.f32, shape=(self.max_n))
        self.x = trail_x
        self.y = trail_y
        self.substep = substep
        self.trail = Pixels(self.x, self.y, evaporate=diffuse_amount, render=False)
        self.init()
    @ti.kernel
    def init(self):
        self.init_rules()
        self.init_angle()
    @ti.func
    def init_angle(self):
        for i in range(self.max_n):
            self.heading[i] = ti.random(ti.f32) * 2.0 * ti.math.pi
    @ti.func
    def init_rules(self):
        for i in range(self.species_n):
            self.rules[i] = PhysarumParams(
                sense_angle = ti.random(ti.f32) * 0.3 * ti.math.pi,
                sense_dist  = ti.random(ti.f32) * 100.0 + 0.1,
                move_angle  = ti.random(ti.f32) * 0.3 * ti.math.pi,
                move_dist   = ti.random(ti.f32) * 4.0 + 0.1)
    @ti.kernel
    def move(self, field: ti.template()):
        for i in range(self.max_n):
            if field[i].active > 0.0:
                p = field[i]
                rule = self.rules[p.species]
                pos, ang = p.pos, self.heading[i]
                c = self.sense(pos, ang,                    rule.sense_dist).norm()
                l = self.sense(pos, ang - rule.sense_angle, rule.sense_dist).norm()
                r = self.sense(pos, ang + rule.sense_angle, rule.sense_dist).norm()
                if l < c < r:
                    ang += rule.move_angle
                elif l > c > r:
                    ang -= rule.move_angle
                elif r > c and c < l:
                    ang += rule.move_angle * (2 * (ti.random() < 0.5) - 1) # TODO: magic numbers
                pos += ti.Vector([ti.cos(ang), ti.sin(ang)]) * rule.move_dist
                field[i].pos = pos
                self.heading[i] = ang
    @ti.func
    def sense(self, pos, ang, dist):
        # TODO: speciate based on rgba compared with field[i].rgba
        p = pos + ti.Vector([ti.cos(ang), ti.sin(ang)]) * dist
        px = ti.cast(p[0], ti.i32) % self.x
        py = ti.cast(p[1], ti.i32) % self.y
        return self.trail.px.rgba[px, py]
    @ti.kernel
    def deposit(self, field: ti.template()):
        for i in range(self.max_n):
            if field[i].active > 0.0:
                p = field[i]
                x = ti.cast(p.pos[0], ti.i32) % self.x
                y = ti.cast(p.pos[1], ti.i32) % self.y
                self.trail.px.rgba[x, y] += p.rgba# * p.mass
    def step(self, field):
        self.deposit(field)
        self.trail.diffuse()
    def process(self, field):
        for i in range(self.substep):
            self.move(field)
            self.step(field)
    def reset(self):
        self.init()
    def __call__(self, field):
        self.process(field)
        return self.trail.px.rgba

def main(host="127.0.0.1", port=4000):
    seed = int(time.time())
    ti.init(arch=ti.vulkan, random_seed=seed)
    # ti.init(random_seed=seed)
    osc = OSC(host, port, verbose=False, concurrent=True)
    fps = 120
    x = 1920
    y = 1080
    n = 8192
    species = 5
    particles = Particles(x, y, n, species)
    pixels = Pixels(x,y, fps=fps)
    physarum = Physarum(x, y, n, species)

    def reset():
        particles.reset()
        pixels.reset()
        physarum.reset()
    update = Updater(reset, fps*4)

    osc_update = OSCUpdaters(osc, client="particles",
        receives={
            "/tolvera/physarum/reset":   reset,
            "/tolvera/physarum/set/pos": particles.osc_set_pos,
            "/tolvera/physarum/set/vel": particles.osc_set_vel
        }, receive_count=10,
        sends={
            "/tolvera/physarum/get/pos/all": particles.osc_get_pos_all
        }, send_count=60
    )

    def render():
        # osc_update() 
        update()
        physarum(particles.field)
        pixels.set(physarum.trail.px)

    pixels.show(render)

if __name__ == '__main__':
    run(main())
