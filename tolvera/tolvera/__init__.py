'''
TODO: test re-initialising in REPL
'''

from typing import Any
from .utils import *
from .consts import *
from .particles import *
from .species import *
from .pixels import *
from .vera import *
from .cv import *
from .patches import *
from .rules import *

class Tolvera:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.init()
    def init(self):
        self.o  = init(**self.kwargs)
        self.s  = Species(self.o)
        self.px = Pixels(self.o, self.s)
        self.p  = Particles(self.o, self.s, self.px)
        self.flock = vera.Flock(self.o.species)
    def randomise(self):
        self.s.randomise()
        self.p.randomise()
        self.flock.randomise()
    def reset(self, **kwargs):
        if kwargs is not None:
            self.kwargs = kwargs
        self.init()
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
    def render(self, func=None, **kwargs):
        render(func, self.px, **kwargs)
