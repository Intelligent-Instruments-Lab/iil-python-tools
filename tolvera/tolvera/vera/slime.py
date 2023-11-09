'''
FIXME: deposit: bug in species indexing causing color changes
TODO: sense_rgba: 
    test
    remove `a` from `rgba`?
    separate slime class? slimergba?
'''

import taichi as ti
from ..pixels import Pixels
from ..state import State
from ..utils import Options, CONSTS

@ti.data_oriented
class Slime:
    def __init__(self, options: Options, evaporate: ti.f32 = 0.99):
        self.o = options
        self.CONSTS = CONSTS({
            'SENSE_ANGLE': (ti.f32, ti.math.pi * 0.3),
            'MOVE_ANGLE':  (ti.f32, ti.math.pi * 0.3),
            'SUBSTEP':     (ti.i32, 1)
        })
        self.particles = State(self.o, {
            'sense_angle':  (0.,10.),
            'sense_left':   (0.,10.),
            'sense_centre': (0.,10.),
            'sense_right':  (0.,10.),
        }, self.o.p, osc=('get'), name='slime_particles', randomise=False)
        self.species = State(self.o, {
            'sense_angle': (0., 1.),
            'sense_dist':  (0., 50.),
            'move_angle':  (0., 1.),
            'move_dist':   (0., 4.),
            'evaporate':   (0., 1.)
        }, self.o.s, osc=('set'), name='slime_species')
        self.trail = Pixels(self.o)
        self.evaporate = ti.field(dtype=ti.f32, shape=())
        self.evaporate[None] = evaporate
    def randomise(self):
        self.species.randomise()
        self.particles.randomise()
    @ti.kernel
    def move(self, field: ti.template()):
        for i in range(field.shape[0]):
            if field[i].active == 0.0: continue

            p    = field[i]
            ang  = self.particles.field[i,0].sense_angle
            species = self.species.field[p.species, p.species] # diagonal

            sense_angle = species.sense_angle * self.CONSTS.SENSE_ANGLE
            move_angle  = species.move_angle  * self.CONSTS.MOVE_ANGLE
            
            c = self.sense(p.pos, ang, species.sense_dist).norm()
            l = self.sense(p.pos, ang - sense_angle, species.sense_dist).norm()
            r = self.sense(p.pos, ang + sense_angle, species.sense_dist).norm()
            
            if l < c < r:
                ang += move_angle
            elif l > c > r:
                ang -= move_angle
            elif r > c and c < l:
                # TODO: magic numbers, move to @ti.func inside utils?
                ang += move_angle * (2 * (ti.random() < 0.5) - 1)
            
            p.pos += ti.Vector([ti.cos(ang), ti.sin(ang)]) \
                * species.move_dist * p.active
            
            self.particles.field[i,0].sense_angle  = ang
            self.particles.field[i,0].sense_centre = c
            self.particles.field[i,0].sense_left   = l
            self.particles.field[i,0].sense_right  = r
            field[i].pos = p.pos
    @ti.func
    def sense(self, pos, ang, dist):
        p = pos + ti.Vector([ti.cos(ang), ti.sin(ang)]) * dist
        px = ti.cast(p[0], ti.i32) % self.o.x
        py = ti.cast(p[1], ti.i32) % self.o.y
        pixel = self.trail.px.rgba[px, py]
        return pixel
    @ti.func
    def sense_rgba(self, pos, ang, dist, rgba):
        p = pos + ti.Vector([ti.cos(ang), ti.sin(ang)]) * dist
        px = ti.cast(p[0], ti.i32) % self.o.x
        py = ti.cast(p[1], ti.i32) % self.o.y
        px_rgba = self.trail.px.rgba[px, py]
        px_rgba_weighted = px_rgba * (1.0 - (px_rgba - rgba).norm())
        return px_rgba_weighted
    @ti.kernel
    def deposit(self, particles: ti.template(), species: ti.template()):
        for i in range(particles.shape[0]):
            if particles[i].active == 0.0: continue
            p, s = particles[i], species[particles[i].species, 0]
            x = ti.cast(p.pos[0], ti.i32) % self.o.x
            y = ti.cast(p.pos[1], ti.i32) % self.o.y
            rgba = ti.Vector([s.r, s.g, s.b, s.a])
            self.trail.circle(x, y, p.size, rgba * p.active)
    def step(self, particles):
        for i in range(self.CONSTS.SUBSTEP):
            self.move(particles.field)
            self.deposit(particles.field, particles.species.field)
            self.trail.diffuse(self.evaporate[None])
    def __call__(self, particles):
        self.step(particles)
        return self.trail.px
