'''
TODO: Test Rule struct as attribute, switch to dict->ti.types.struct(att=type)?
TODO: Indicate which methods need to be implemented by subclasses
TODO: species_rule(s), species_rules_from_list
TODO: Canvas wrapping default methods
TODO: Add default main?
'''

from typing import Any
import taichi as ti

from tolvera.consts import VERA as VERA_CONSTS

@ti.dataclass
class VeraRules:
    test: ti.f32

@ti.data_oriented
class Vera:
    def __init__(self,
                 x: ti.i32,
                 y: ti.i32,
                 species: ti.i32,
                 rules = None) -> None:
        assert rules is not None, f"[Tölvera] Error: Missing Rules for {self.__class__.__name__}"
        self.x = x
        self.y = y
        self.species_n = species
        self._rules_struct = rules
        self._rules = rules.field(shape=(species,species))
        self.consts = VERA_CONSTS
        self.init()
    @ti.kernel
    def init(self):
        for i, j in ti.ndrange(self.species_n, self.species_n):
            self._rules[i,j] = self._rules_struct(ti.random(ti.f32)*self.consts.TEST)
    def reset(self):
        self.init()
    @ti.kernel
    def step(self, species: ti.template(), particles: ti.template(), pixels: ti.template()):
        for i, j in ti.ndrange(self.species_n, self.species_n):
            # if species[i].active > 0.0 and species[j].active > 0.0:
            if particles[i].active > 0.0:
                self.render(pixels)
    @ti.func
    def render(self, pixels: ti.template()):
        pass
    def __call__(self, species, particles, pixels) -> Any:
        self.step(species, particles, pixels)
    def rule(self, i: ti.i32, j: ti.i32, rule=None):
        if rule is None:
            return self.rules[i,j]
        else:
            assert type(rule) == self._rules_struct, f"[Tölvera] Error: Rule must be of type {self._rules_struct}"
            self.rules[i,j] = rule
    def rules(self, rules: ti.template()=None):
        if rules is None:
            return self._rules
        else:
            self.rules = rules
    @ti.kernel
    def rules_from_list(self, rules: ti.types.ndarray()):
        assert rules.shape[0] == self.species_n * self.species_n * 4, f"rules.shape[0]={rules.shape[0]} != {self.species_n * self.species_n * 4}"
        for i in range(self.species_n):
            self._rules_from_list(i, rules)
    @ti.func
    def _rules_from_list(self, i, rules: ti.types.ndarray()):
        # c = self.consts
        for j in range(self.species_n):
            self._rules[i,j] = self._rules_struct(rules[i*4+j+0])


