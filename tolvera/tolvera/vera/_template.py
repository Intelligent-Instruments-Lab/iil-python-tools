'''
'''

import taichi as ti
from ..state import State
from ..utils import Options, CONSTS

@ti.data_oriented
class ParticleTemplate:
    def __init__(self, options: Options):
        self.o = options
        self.state = State(self.o, {
            'state': (.01, 1.),
        }, self.o.s, osc=('set'), name='state')
    def randomise(self):
        self.state.randomise()
    @ti.kernel
    def step(self, particles: ti.template()):
        n = particles.shape[0]
        for i in range(n):
            pass
    def __call__(self, particles):
        self.step(particles.field)
