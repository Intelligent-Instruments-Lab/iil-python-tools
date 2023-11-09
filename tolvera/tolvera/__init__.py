'''
TODO: test reset
'''

from typing import Any
from .utils import *
from .particles import *
from .pixels import *
from .vera import *
from .cv import *
from .patches import *
from .state import *

class Tolvera:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.init()
    def init(self):
        self.o  = init(**self.kwargs)
        self.px = Pixels(self.o)
        self.p  = Particles(self.o, self.px)
        self.flock = vera.Flock(self.o)
        self.slime = vera.Slime(self.o)
        if self.o.osc is not False:
            self.add_to_osc_map()
    def randomise(self):
        self.p.randomise()
        self.p.species.randomise()
        self.flock.randomise()
        self.slime.randomise()
    def reset(self, **kwargs):
        if kwargs is not None:
            self.kwargs = kwargs
        self.init()
    def add_to_osc_map(self):
        setter_name = f"{self.o.name_clean}_set"
        getter_name = f"{self.o.name_clean}_get"
        self.o.osc_map.receive_args_inline(setter_name+'_randomise', self.randomise)
        self.o.osc_map.receive_args_inline(setter_name+'_particles_randomise', self.p._randomise)
    def render(self, func=None, **kwargs):
        render(func, self.px, **kwargs)
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
