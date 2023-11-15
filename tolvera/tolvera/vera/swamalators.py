'''
Based on https://www.complexity-explorables.org/explorables/swarmalators/
'''

'''
TODO: test
TODO: 
'''

'''
@ti.dataclass
class Swarmalator:
    pos:    ti.math.vec2
    dpos:   ti.math.vec2
    vel:    ti.math.vec2
    omega:  ti.f32
    domega: ti.f32
    theta:  ti.f32
    dtheta: ti.f32
'''

import taichi as ti
from ..state import State
from ..utils import Options, CONSTS

@ti.data_oriented
class Swamalators:
    def __init__(self, options: Options):
        self.o = options
        self.CONSTS = CONSTS({
            'dt':     .1,
            'dtt':    ti.math.sqrt(.1),
            'radius': 4.,
            'L':      1.3,
            'two_L':  2.6,
            'omega0': 1.,
            'two_pi': ti.math.pi * 2,
        })
        self.particles = State(self.o, {
            'x': (0., 1.),
            'y': (0., 1.),
            'vx': (0., 1.),
            'vy': (0., 1.),
            'dx': (0., 1.),
            'dy': (0., 1.),
            'omega': (0., 1.),
            'domega': (0., 1.),
            'theta': (0., 1.),
            'dtheta': (0., 1.),
            'r': (0., 1.),
            'g': (0., 1.),
            'b': (0., 1.),
        }, self.o.particles, name='swamalators_particles', randomise=False)
        self.controls = State(self.o, {
            'J': (-1., 1.), # coupling strength
            'J': (-1., 1.), # synchronisation strength
            'noise': (.0, 5.), # wiggle
            'omega': (.01, 1.), # frequency variation
            'speed': (.01, 1.), # normal speed
        }, self.o.s, osc=('set'), name='swamalators_state', randomise=False)
        self.presets = [
            {'J': 1.0,  'K': 0.51,  'noise': 0.0, 'omega': 0.4439}, # Makes Me Dizzy
            {'J': 0.5,  'K': 0.0,   'noise': 0.0, 'omega': 0.0}, # Rainbow Ring
            {'J': 0.93, 'K': -0.88, 'noise': 0.0, 'omega': 0.0}, # Dancing Circus
            {'J': 0.1,  'K': 1.0,   'noise': 0.0, 'omega': 0.0}, # Uniform Blob
            {'J': 0.1,  'K': 1.0,   'noise': 0.0, 'omega': 0.8}, # Solar Convection
            {'J': 1.0,  'K': -0.12, 'noise': 0.0, 'omega': 0.0}, # Fractions
        ]
        self.i = 0
        self.phasemod = 1
        self.init()
    def init(self):
        self.controls.field[0,0] = self.presets[0]
        self.reset()
    def randomise(self):
        self.state.randomise()
    @ti.kernel
    def reset(self):
        CONSTS = self.CONSTS
        self.i = 0
        mvx = 0.
        mvy = 0.
        for i in range(self.o.particles):
            knut = CONSTS.two_pi * ti.random(ti.f32)
            self.particles.field[i,0] = self.particles.struct({
                'x':      CONSTS.two_L * (ti.random(ti.f32) - 0.5),
                'y':      CONSTS.two_L * (ti.random(ti.f32) - 0.5),
                'vx':     ti.math.cos(knut),
                'vy':     ti.math.sin(knut),
                'dx':     0.,
                'dy':     0.,
                'omega':  CONSTS.omega0,
                'domega': ti.randn(),
                'theta':  CONSTS.two_pi * ti.random(ti.f32),
                'dtheta': 0.,
                'color':  0.,
            })
            mvx += self.particles.field[i,0].vx
            mvy += self.particles.field[i,0].vy
        mvx /= self.o.particles
        mvy /= self.o.particles
        for i in range(self.o.particles):
            self.particles.field[i,0].vx -= mvx
            self.particles.field[i,0].vy -= mvy
    @ti.kernel
    def perturb(self):
        CONSTS = self.CONSTS
        for i in range(self.o.particles):
            p = self.particles.field[i,0]
            w = CONSTS.two_pi * ti.random(ti.f32)
            p.x += CONSTS.radius * ti.math.cos(w)
            p.y += CONSTS.radius * ti.math.sin(w)
            p.theta += ti.math.pi * (ti.random(ti.f32) - 0.25 * CONSTS.two_pi)
            self.particles.field[i,0] = p
    @ti.kernel
    def step(self, particles: ti.template()):
        self.i += 1
        pn = self.o.particles
        c = self.controls.field[0,0]
        CONSTS = self.CONSTS
        for i in range(pn):
            p1 = self.particles.field[i,0]
            p1.dx = p1.vx * c.speed
            p1.dy = p1.vy * c.speed
            p1.dtheta = p1.omega * p1.domega * c.omega * self.phasemod 
            for j in range(pn):
                if i == j: continue
                p2 = self.particles.field[j,0]
                d = ti.Vector([p2.x - p1.x, p2.y - p1.y])
                kernel = (1+c.J * ti.math.cos(p2.theta - p1.theta)/d - 1.0/(d*d))/pn
                p1.dx = (p2.x - p1.y) * kernel
                p1.dy = (p2.y - p1.y) * kernel
                p1.dtheta = c.K / pn * ti.math.sin(p2.theta - p1.theta) / d
            p1.dx     *= CONSTS.dt
            p1.dy     *= CONSTS.dt
            p1.dtheta *= CONSTS.dt
            p1.x += p1.dx + CONSTS.dtt * c.noise * (ti.random(ti.f32)-0.5)
            p1.y += p1.dy + CONSTS.dtt * c.noise * (ti.random(ti.f32)-0.5)
            p1.theta += p1.dtheta
            p1.color = p1.theta/2/ti.math.pi
            self.particles.field[i,0] = p1
            particles.field[i].pos = ti.Vector([self.X(p1.x), self.Y(p1.y)])
    @ti.func
    def X(self, x):
        return (x + self.CONSTS.L) / self.CONSTS.two_L * self.o.x - self.o.x/2
    @ti.func
    def Y(self, y):
        return (y + self.CONSTS.L) / self.CONSTS.two_L * self.o.y - self.o.y/2
    def __call__(self, particles):
        self.step(particles.field)
