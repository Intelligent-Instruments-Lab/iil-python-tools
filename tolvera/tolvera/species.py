'''
TODO: prototype
'''

import taichi as ti
import numpy as np
from .consts import SPECIES as SPECIES_CONSTS
from .utils import Options

vec1 = ti.types.vector(1, ti.f32)
vec2 = ti.math.vec2
vec3 = ti.math.vec3
vec4 = ti.math.vec4

'''
TODO: Compare with ParticlesParams and CONSTS
'''
@ti.dataclass
class SpeciesParams:
    i:         ti.i32
    n:         ti.i32
    size:      ti.f32
    max_speed: ti.f32
    mass:      ti.f32
    rgba:      vec4
    active:    ti.f32
    diffusion: ti.f32 # for pixels

@ti.data_oriented
class Species:
    def __init__(self, options: Options):
        self.o = options
        self.species = SpeciesParams.field(shape=(self.o.n))
        self.particles_per_species = self.o.n // self.o.species
        self.consts = SPECIES_CONSTS()
    def init(self):
        self.randomise()
    @ti.kernel
    def randomise(self):
        for i in range(self.o.species):
            self.species[i].rgba = [ti.random(ti.f32),ti.random(ti.f32),ti.random(ti.f32),1]
        c = self.consts
        for i in range(self.max_n):
            s = i % self.species_n
            self.species[i].i         = s
            self.species[i].size      = c.SIZE_MIN      + c.SIZE_SCALE      * ti.random(ti.f32)
            self.species[i].speed     = c.SPEED_MIN     + c.SPEED_SCALE     * ti.random(ti.f32)
            self.species[i].max_speed = c.MAX_SPEED_MIN + c.MAX_SPEED_SCALE * ti.random(ti.f32)
            self.species[i].mass      = c.MASS_MIN      + c.MASS_SCALE      * ti.random(ti.f32)
            self.species[i].decay     = c.DECAY_MIN     + c.DECAY_SCALE     * ti.random(ti.f32)
            self.species[i].active    = 1.0
    def active(self, i: ti.i32=None):
        if i is None:
            return self.species[i].active
        return self.species[i].active > 0.0
    def __call__(self):
        pass
    