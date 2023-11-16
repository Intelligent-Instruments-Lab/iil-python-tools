'''
TODO: test reset
TODO: add attractors
TODO: combined OSC setter(s) for species+flock+slime+rd etc..?
TODO: global scalars for speed, i: global frame index???
TODO: async runner based on sardine @s[wim?
TODO: implement reset()
TODO: particles_per_species only due to way species are allocated to particles
TODO: turn Options into a Singleton?
    decouple tolvera globals from taichi globals first?
TODO: move init funcs to __init__.py? re namespace/decorators
        `@tv.cleanup` `@tv.render`
'''

from typing import Any
from iipyper.state import _lock
from sys import exit

from .patches import *
from .utils import *
from .particles import *
from .pixels import *
from .vera import Vera
from ._taichi import Taichi
from ._osc import OSC
from ._iml import IML
from ._cv import CV

class Tolvera:
    def __init__(self, **kwargs) -> None:
        self.setup(**kwargs)
        self.init(**kwargs)
    def init(self, **kwargs):
        self.ti = Taichi(self, **kwargs)
        self.osc = OSC(self, **kwargs)
        self.iml = IML(self, **kwargs)
        # self.cv = CV(self, **kwargs)
    def setup(self, **kwargs):
        self.kwargs    = kwargs
        self.x         = kwargs.get('x', 1920)
        self.y         = kwargs.get('y', 1080)
        self.particles = kwargs.get('particles', 1024)
        self.species   = kwargs.get('species', 4)
        self.substep   = kwargs.get('substep', 1)
        self.evaporate = kwargs.get('evaporate', 0.95)
        self.p_per_s = self.particles // self.species
        self.p = Particles(self)
        self.s = self.p.species
        self.v = Vera(self)
        self.px = Pixels(self)
        self._cleanup_fns = []
        if self.osc is not None:
            self.add_to_osc_map()
    def randomise(self):
        self.p.randomise()
        self.s.randomise()
        self.v.randomise()
    def reset(self, **kwargs):
        if kwargs is not None:
            self.kwargs = kwargs
        self.setup()
    def add_to_osc_map(self):
        setter_name = f"{self.o.name_clean}_set"
        getter_name = f"{self.o.name_clean}_get"
        self.osc.map.receive_args_inline(setter_name+'_randomise', self.randomise)
        self.osc.map.receive_args_inline(setter_name+'_particles_randomise', self.p._randomise)
    def run(self, f, px, **kwargs):
        while self.ti.window.running:
            with _lock:
                if f is not None: f(**kwargs)
                if self.osc is not False: self.osc.osc_map()
                self.ti.show(px)
    def stop(self):
        print(f"\n[Tölvera] Exiting {self.name}...")
        for f in self._cleanup_fns: f()
        exit(0)
    def render(self, f=None, px=None, **kwargs):
        try: self.run(f, px, **kwargs)
        except KeyboardInterrupt: self.stop()
    def cleanup(self, f=None):
        """
        @cleanup decorator based on iipyper
        make functions run on KeyBoardInterrupt (before exit)
        cleanup functions must be defined before render is called!
        """
        def decorator(f):
            self._cleanup_fns.append(f)
            return f
        if f is None: # return a decorator
            return decorator
        else: #bare decorator case; return decorated function
            return decorator(f)
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.render(*args, **kwds)

'''
# Render as decorator
def render(px=None, **kwargs):
    def decorator(func):
        def wrapper(*args, **func_kwargs):
            try:
                # add any pre-function call logic
                _run(func, px, **{**kwargs, **func_kwargs})
            except KeyboardInterrupt:
                _stop()
            # add any post-function call logic
        return wrapper

    return decorator
'''

