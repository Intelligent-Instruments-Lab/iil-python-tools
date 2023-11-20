import taichi as ti

from .state import State

class Species:
    def __init__(self, tolvera, **kwargs) -> None:
        self.tv = tolvera
        self.kwargs = kwargs
        self.n = self.tv.species
        self.state = State(self.tv, {
            'size':  (ti.f32, 1., 4.),
            'speed': (ti.f32, 0., 4.),
            'mass':  (ti.f32, 0., 1.),
            'decay': (ti.f32, .9, .999),
            'rgba':  (ti.math.vec4, 0., 1.),
        }, shape=(self.n, self.n), osc=('set'), name='species')
    def randomise(self):
        self.state.randomise()