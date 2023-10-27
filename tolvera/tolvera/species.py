'''
TODO: Why do particle positions change when (re-)randomising species?
'''

import taichi as ti
import numpy as np
from .consts import SPECIES as SPECIES_CONSTS
from .utils import Options

vec2 = ti.math.vec2
vec3 = ti.math.vec3
vec4 = ti.math.vec4

# Attributes of a species
@ti.dataclass
class SpeciesField:
    id:        ti.i32
    active:    ti.f32
    max_size:  ti.f32
    max_speed: ti.f32
    max_mass:  ti.f32
    max_decay: ti.f32
    rgba:      vec4
    # g:         vec1
    # rgb:       vec3
    # n:         ti.i32
    # diffusion: ti.f32 # for pixels

@ti.data_oriented
class Species:
    def __init__(self, options: Options):
        self.o = options
        self.field = SpeciesField.field(shape=(self.o.species))
        self.consts = SPECIES_CONSTS
        self.init()
    def init(self):
        self.assign_species()
        self.randomise()
    @ti.kernel
    def assign_species(self):
        for i in range(self.o.n):
            self.field[i].id = i % self.o.species
    @ti.kernel
    def randomise(self):
        for i in range(self.o.species):
            self.field[i].rgba = [ti.random(ti.f32),ti.random(ti.f32),ti.random(ti.f32),1]
        c = self.consts
        for i in range(self.o.n):
            self.field[i].active    = 1.0
            self.field[i].max_size  = c.SIZE_MIN      + c.SIZE_SCALE      * ti.random(ti.f32)
            self.field[i].max_speed = c.MAX_SPEED_MIN + c.MAX_SPEED_SCALE * ti.random(ti.f32)
            self.field[i].max_mass  = c.MASS_MIN      + c.MASS_SCALE      * ti.random(ti.f32)
            self.field[i].max_decay = c.DECAY_MIN     + c.DECAY_SCALE     * ti.random(ti.f32)
    def active(self, i: ti.i32=None):
        if i is None:
            return self.field[i].active
        return self.field[i].active > 0.0
    def __call__(self):
        pass
    