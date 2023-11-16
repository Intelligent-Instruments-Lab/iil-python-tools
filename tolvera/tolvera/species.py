import taichi as ti

from .state import State

class Species:
    def __init__(self, tolvera, **kwargs) -> None:
        self.tv = tolvera
        self.kwargs = kwargs
        self.n = self.tv.species
        self.state = State(self.tv, {
            'size':   (1., 4.),
            'speed':  (0., 4.),
            'mass':   (0., 1.),
            'decay':  (.9, .999),
            'r':      (0., 1.),
            'g':      (0., 1.),
            'b':      (0., 1.),
            'a':      (1., 1.),
        }, self.n, osc=('set'), name='species')
