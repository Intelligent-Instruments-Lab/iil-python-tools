'''
TODO: test reset
TODO: add attractors
TODO: combined OSC setter(s) for species+flock+slime+rd etc..?
TODO: async runner based on sardine @swim?
TODO: should _packages be singletons?
'''

# External packages
from iipyper.state import _lock
import fire
from sys import exit

# Wrapped external packages
from ._taichi import Taichi
from ._osc import OSC
from ._iml import IML
# from ._cv import CV

# Tölvera components
from .patches import *
from .utils import *
from .particles import *
from .pixels import *
from .vera import Vera

class Tolvera:
    def __init__(self, **kwargs) -> None:
        """
        Initialize and setup Tölvera with given keyword arguments.

        Args:
            **kwargs: Keyword arguments for setup and initialization.
        """
        self.kwargs = kwargs
        self.init(**kwargs)
        self.setup(**kwargs)
        print(f"[Tölvera] Initialization and setup complete.")
    def init(self, **kwargs):
        """
        Initialize external packages with given keyword arguments.
        This only happens once when Tölvera is first initialized.

        Args:
            **kwargs: Keyword arguments for component initialization.
        """
        self.i = 0
        self.x = kwargs.get('x', 1920)
        self.y = kwargs.get('y', 1080)
        self.name = kwargs.get('name', 'Tölvera')
        self.name_clean = clean_name(self.name)
        self.ti = Taichi(self, **kwargs)
        self.osc = OSC(self, **kwargs)
        self.iml = IML(self, **kwargs)
        # self.cv = CV(self, **kwargs)
        if self.osc.osc is not False: # TODO: check
            self.add_to_osc_map()
        print(f"[Tölvera] Initialization complete.")
    def setup(self, **kwargs):
        """
        Setup Tölvera with given keyword arguments.
        This can be called multiple throughout the lifetime of Tölvera.

        Args:
            **kwargs: Keyword arguments for setup.
        """
        self.particles = kwargs.get('particles', 1024)
        self.species   = kwargs.get('species', 4)
        self.p_per_s   = self.particles // self.species
        self.substep   = kwargs.get('substep', 1)
        self.evaporate = kwargs.get('evaporate', 0.95)
        self.px = Pixels(self, **kwargs)
        self.s = Species(self, **kwargs)
        self.p = Particles(self, self.s, **kwargs)
        self.v = Vera(self, **kwargs)
        self._cleanup_fns = []
        print(f"[Tölvera] Setup complete.")
    def randomise(self):
        """
        Randomize particles, species, and Vera.
        """
        self.p.randomise()
        self.s.randomise()
        self.v.randomise()
    def reset(self, **kwargs):
        """
        Reset Tölvera with given keyword arguments.
        This will call setup() with given keyword arguments, but not init().

        Args:
            **kwargs: Keyword arguments for reset.
        """
        if kwargs is not None:
            self.kwargs = kwargs
        self.setup()
    def add_to_osc_map(self):
        """
        Add top-level Tölvera functions to OSC map.
        """
        setter_name = f"{self.o.name_clean}_set"
        getter_name = f"{self.o.name_clean}_get"
        self.osc.map.receive_args_inline(setter_name+'_randomise', self.randomise)
        self.osc.map.receive_args_inline(setter_name+'_reset', self.reset) # TODO: kwargs?
        self.osc.map.receive_args_inline(setter_name+'_particles_randomise', self.p._randomise) # TODO: move inside Particles
    def run(self, f, px, **kwargs):
        """
        Run Tölvera with given function, pixels, and keyword arguments.

        Args:
            f: Function to run.
            px: Pixels to show.
            **kwargs: Keyword arguments for function.
        """
        while self.ti.window.running:
            with _lock:
                if f is not None: f(**kwargs)
                if self.osc is not False: self.osc.osc_map()
                self.ti.show(px)
                self.i += 1
    def stop(self):
        """
        Stop Tölvera and exit.
        """
        print(f"\n[Tölvera] Exiting {self.name}...")
        for f in self._cleanup_fns: f()
        exit(0)
    def render(self, px=None, **kwargs):
        """
        Decorator for rendering Tölvera with given pixels and keyword arguments.

        Args:
            px: Pixels to render.
            **kwargs: Keyword arguments for rendering.

        Returns:
            Decorator function.
        """
        def decorator(func):
            def wrapper(*args, **func_kwargs):
                try:
                    self.run(func, px, **{**kwargs, **func_kwargs})
                except KeyboardInterrupt:
                    self.stop()
            return wrapper
        print('render', decorator, px, kwargs)
        return decorator
    def cleanup(self, f=None):
        """
        Decorator for cleanup functions based on iipyper.
        Make functions run on KeyBoardInterrupt (before exit).
        Cleanup functions must be defined before render is called!

        Args:
            f: Function to cleanup.

        Returns:
            Decorator function if f is None, else decorated function.
        """
        def decorator(f):
            self._cleanup_fns.append(f)
            return f
        if f is None: # return a decorator
            return decorator
        else: #bare decorator case; return decorated function
            return decorator(f)


def main(**kwargs):
    tv = Tolvera(**kwargs)

    @tv.render
    def _():
        tv.p()

if __name__ == '__main__':
    fire.Fire(main)
