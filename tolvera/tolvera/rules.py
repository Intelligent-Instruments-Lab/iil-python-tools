'''
TODO: 1D, 2D, ND versions of rules?
TODO: add setters for args, kwargs, list
TODO: should setters be normalised and scaled via self.dict?
TODO: randomise_by_key() (randomise individual rules)
TODO: types support for more than just ti.f32, e.g ti.math.vec2?
TODO: should setters be @ti.func struct methods? https://docs.taichi-lang.org/docs/dataclass#notes
TODO: add endpoints to OSCMap when passed
TODO: add IML mapping?
TODO: add Sardine patterns?
TODO: add arbitrary state container(s) of shape=(particles)
TODO: update species n when Tolvera is re-initialised
'''

from typing import Any
import taichi as ti
from .utils import get_field_attr

@ti.data_oriented
class Rules:
    def __init__(self, rules: dict[str, tuple[ti.f32, ti.f32]], species: int):
        self.dict = rules
        self.species = species
        self.len = len(rules)
        self.len_ndarr_all = len(rules)*self.species*self.species
        self.len_ndarr_species = len(rules)*self.species
        self.struct = ti.types.struct(**{k: ti.f32 for k,v in self.dict.items()})
        self.field = self.struct.field(shape=(self.species, self.species))
        self.init()

    def init(self):
        self.randomise()

    @ti.kernel
    def randomise(self):
        for i, j in ti.ndrange(self.species, self.species):
            rules = {k: v[0] + (v[1] - v[0]) * ti.random(ti.f32)
                     for k, v in self.dict.items()}
            self.field[i, j] = self.struct(**rules)

    def set_args(self, index: tuple, *rules: Any):
        self.field[index] = self.struct(*rules)

    def set_kwargs(self, index: tuple, **rules: Any):
        for k, v in self.dict.items():
            if k not in rules:
                rules[k] = get_field_attr(self.field, k, index)
        self.field[index] = self.struct(**rules)

    def set_list(self, index: tuple, rules: list):
        self.field[index] = self.struct(*rules)

    def set_all_ndarray(self, rules: ti.types.ndarray(dtype=ti.f32, ndim=3)):
        '''
        Should be @ti.kernel, but can't do `*rules[i,j]`

        Args:
            rules: np.array((species, species, rules), dtype=np.float32)
        '''
        for i, j in ti.ndrange(rules.shape[0], rules.shape[0]):
            self.field[i,j] = self.struct(*rules[i,j])

    def set_all_list(self, rules: list):
        '''
        Flat list of rules for each species pair

        Args:
            rules = [i0j0r0-i0j0rN, i0j1r0-i0j1rN, i1j1r0-i1jrN, ...]
        '''
        assert len(rules) == self.len_ndarr_all, f"len(rules) != len_ndarr_all ({len(rules)} != {self.len_ndarr_all})"
        l = self.len
        for i, j in ti.ndrange(self.species, self.species):
            rule = [rules[i*l+j+r] for r in range(l)]
            self.field[i,j] = self.struct(*rule)

    def set_species_ndarray(self, i: ti.i32, rules: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        '''
        Args:
            rules: np.array((species, rules), dtype=np.float32)
        '''
        for j in range(self.species):
            self.field[i,j] = self.struct(*rules[j])

    def set_species_list(self, i: ti.i32, rules: list):
        '''
        Args:
            rules = [i0r0-i0rN, i1r0-i1rN, ...]
        '''
        assert len(rules) == self.len_ndarr_species, f"len(rules) != len_ndarr_species ({len(rules)} != {self.len_ndarr_species})"
        l = self.len
        for j in range(self.species):
            rule = [rules[j*l+r] for r in range(l)]
            self.field[i,j] = self.struct(*rule)
