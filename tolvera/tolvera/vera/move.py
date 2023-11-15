import taichi as ti
from ..state import State
from ..utils import Options, CONSTS

@ti.data_oriented
class Move:
    def __init__(self, options: Options):
        self.o = options
        pass
    @ti.kernel
    def step(self, particles: ti.template()):
        for i in range(particles.shape[0]):
            if particles[i].active == 0: continue
            p1 = particles[i]
            particles[i].pos += particles[i].vel * p1.speed * p1.active
    def __call__(self, particles):
        self.step(particles.field)
