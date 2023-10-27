import taichi as ti
from ..rules import Rules

@ti.data_oriented
class Flock:
    def __init__(self, species):
        self.species = species
        self.rules = Rules({
            'separate': {'type': ti.f32, 'min': 0., 'max': 1.},
            'align':    {'type': ti.f32, 'min': 0., 'max': 1.},
            'cohere':   {'type': ti.f32, 'min': 0., 'max': 1.},
            'radius':   {'type': ti.f32, 'min': 0., 'max': 300.0}
        }, self.species)
    def randomise(self):
        self.rules.randomise()
    @ti.kernel
    def step(self, field: ti.template()):
        n = field.shape[0]
        for i in range(n):
            p1 = field[i]
            if p1.active == 0: continue
            separate = ti.Vector([0., 0.])
            align    = ti.Vector([0., 0.])
            cohere   = ti.Vector([0., 0.])
            nearby = 0
            r = self.rules.struct()
            for j in range(n):
                p2 = field[j]
                if i==j and p2.active == 0: continue
                r = self.rules.field[p1.species, p2.species]
                dis = p1.dist(p2)
                if dis.norm() < r.radius:
                    nearby   += 1
                    separate += dis
                    align    += p2.vel
                    cohere   += p2.pos
            if nearby > 0:
                separate = separate/nearby * p1.active * r.separate
                align    = align/nearby    * p1.active * r.align
                cohere   = (cohere/nearby - p1.pos).normalized() * p1.active * r.cohere
                field[i].vel  += (separate+align+cohere).normalized()
    def __call__(self, field: ti.template()):
        self.step(field)
