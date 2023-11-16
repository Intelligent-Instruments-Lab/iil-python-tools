'''
Draws a membrane around the particles, which attracts/repels them.
Has a centroid
Uses pixels for drawing membrane
Optional debug rendering of self
Wall sticking is actually desirable?
Customisable/overwritten wall behaviours
opencv convex hull?
'''

import taichi as ti
from .state import State
from. pixels import Pixels

@ti.data_oriented
class Membrane:
    def __init__(self) -> None:
        self.walls = State(self.o, {
            'wall_top':    (0., 100.),
            'wall_right':  (0., 100.),
            'wall_bottom': (0., 100.),
            'wall_left':   (0., 100.),
            'turn_top':    (0., 1.),
            'turn_right':  (0., 1.),
            'turn_bottom': (0., 1.),
            'turn_left':   (0., 1.),
        }, 4, osc=True, name='particles_walls', randomise=False)
        # Wall behaviours: top, right, bottom, left (clockwise, a la CSS margin)
        self.wall_margins = ti.field(ti.f32, shape=(4))
        self.turn_factors = ti.field(ti.f32, shape=(4))
        self.wall_margins.fill(50.0)
        self.turn_factors.fill(0.8)
    @ti.func
    def wall_repel(self, i):
        p = self.field[i]
        wt, wr, wb, wl = self.wall_margins[0], self.wall_margins[1], self.wall_margins[2], self.wall_margins[3]
        tt, tr, tb, tl = self.turn_factors[0], self.turn_factors[1], self.turn_factors[2], self.turn_factors[3]
        x, y = p.pos[0], p.pos[1]
        if  (y > self.y-wt): # top
            # self.field.vel[i][1] *= -0.9
            self.field.vel[i][1] -= tt
            self.field.pos[i][1] -= ti.random(ti.f32)*5.0
        elif(x < wr):        # right
            # self.field.vel[i][0] *= -0.9
            self.field.vel[i][0] += tr
            self.field.pos[i][0] += ti.random(ti.f32)*5.0
        elif(y < wb):        # bottom
            # self.field.vel[i][1] *= -0.9
            self.field.vel[i][1] += tb
            self.field.pos[i][1] += ti.random(ti.f32)*5.0
        elif(x > self.x-wl): # left
            # self.field.vel[i][0] *= -0.9
            self.field.vel[i][0] -= tl
            self.field.pos[i][0] -= ti.random(ti.f32)*5.0
    @ti.kernel
    def set_wall_margins(self, top: ti.f32, right: ti.f32, bottom: ti.f32, left: ti.f32):
        self.wall_margins[0] = top
        self.wall_margins[1] = right
        self.wall_margins[2] = bottom
        self.wall_margins[3] = left
    @ti.kernel
    def set_turn_factors(self, top: ti.f32, right: ti.f32, bottom: ti.f32, left: ti.f32):
        self.turn_factors[0] = top
        self.turn_factors[1] = right
        self.turn_factors[2] = bottom
        self.turn_factors[3] = left
    @ti.kernel
    def set_wall(self, wall: ti.i32, margin: ti.f32, turn: ti.f32):
        self.wall_margins[wall] = margin
        self.turn_factors[wall] = turn


