'''
FIXME: still some NaNs happening?
TODO: particle state analysis/applications?
TODO: is half of self.state.dist redundant? print with low n
TODO: re-add credit
'''
import taichi as ti
from ..state import State
from ..utils import CONSTS

@ti.data_oriented
class Flock:
    def __init__(self, tolvera, **kwargs):
        self.tv = tolvera
        self.kwargs = kwargs
        self.particles = State(tolvera, {
            # 'separate': (0., 1.), # ti.math.vec2
            # 'align':    (0., 1.), # ti.math.vec2
            # 'cohere':   (0., 1.), # ti.math.vec2
            'nearby':   (0., self.tv.p.n-1),   # ti.i32
            'dist':     (0., self.tv.x*2), # ti.f32
            'dist_wrap':(0., self.tv.x*2), # ti.f32
        }, self.tv.p.n, osc=('get'), name='flock_particles', randomise=False)
        self.species = State(tolvera, {
            'separate': (.01, 1.),
            'align':    (.01, 1.),
            'cohere':   (.01, 1.),
            'radius':   (.01, 300.)
        }, self.tv.s.n, osc=('set'), name='flock_species')
    def randomise(self):
        self.species.randomise()
    @ti.kernel
    def step(self, particles: ti.template()):
        n = particles.shape[0]
        for i in range(n):
            if particles[i].active == 0: continue
            p1 = particles[i]
            # state = self.particles.field[i,0]
            # state.fill(0.)
            separate = ti.Vector([0., 0.])
            align    = ti.Vector([0., 0.])
            cohere   = ti.Vector([0., 0.])
            nearby   = 0
            species = self.species.struct()
            for j in range(n):
                if i==j and particles[j].active == 0: continue
                p2 = particles[j]
                species = self.species.field[p1.species, p2.species]
                dis_wrap = p1.dist_wrap(p2, self.tv.x, self.tv.y)
                dis_wrap_norm = dis_wrap.norm()
                if dis_wrap_norm < species.radius:
                    separate += dis_wrap
                    align    += p2.vel
                    cohere   += p2.pos
                    nearby   += 1
                self.particles.field[i,j].dist = p1.dist(p2).norm()
                self.particles.field[i,j].dist_wrap = dis_wrap_norm
            if nearby > 0:
                separate = separate/nearby        * p1.active * species.separate
                align    = align/nearby           * p1.active * species.align
                cohere   = (cohere/nearby-p1.pos) * p1.active * species.cohere
                vel      = (separate+align+cohere).normalized()
                particles[i].vel += vel
                particles[i].pos += particles[i].vel * p1.speed * p1.active
            self.particles.field[i,0].nearby = nearby

    def __call__(self, particles):
        self.step(particles.field)
