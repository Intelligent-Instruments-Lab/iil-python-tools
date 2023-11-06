'''
Current implementation:
- State is ti.f32 only
- Shape is int -> (int, int)
    - 1D: use 1st col (i,0)
    - 2D: use (i,j)

TODO: state
    `state: dict[str, tuple[DataType, Any, Any]]`
        Any, Any only needed if state will be controlled by user (species, not particles)
    1 DataType per State, or multiple?
    DataType support?
        ti.i32
        ti.f32
        ti.math.vec2
        ti.math.vec3
        ti.math.vec4
    save/load? serialise/deserialise
TODO: shape
    1D, 2D, ND versions of state?
    update shape when Tolvera is re-initialised?
    `shape: int | tuple[int]`
TODO: randomise
    randomise_by_key() (randomise individual state using ti_field_getattr)
    how to randomise arbitrary shapes with ti.ndrange(**shape), field[...]?
TODO: setters:
    indexing with variable shape?
    self.len* with variable shape?
    finish writing test
TODO: OSC: add endpoints to OSCMap when passed
TODO: IML: add default mapping?
TODO: Sardine: pattern utils?
'''

from typing import Any
import taichi as ti
from .utils import ti_field_getattr, Options
from taichi._lib.core.taichi_python import DataType
from iipyper import OSCMap

@ti.data_oriented
class State:
    def __init__(self, 
                 options: Options,
                 state: dict[str, tuple[ti.f32, ti.f32]], # tuple[DataType, Any, Any]
                 shape: int,# | tuple[int], 
                 randomise: bool = True):
        self.o = options
        self.dict = state
        self.shape = shape
        # self.struct = ti.types.struct(**{k: v[0] for k,v in self.dict.items()})
        # self.field = self.struct.field(shape=self.shape)
        self.struct = ti.types.struct(**{k: ti.f32 for k,v in self.dict.items()})
        self.field = self.struct.field(shape=(self.shape, self.shape))
        self.len = len(state)
        self.len_ndarr_all = len(state)*self.shape*self.shape
        self.len_ndarr_shape = len(state)*self.shape
        self.init(randomise)

    def init(self, randomise: bool = False):
        if randomise: self.randomise()
        if self.o.osc is not None:
            self.add_to_osc_map()

    def add_to_osc_map(self):
        m = self.o.osc_map
        # m.receive_args()(self.set_args)

    def get(self, index: tuple, attribute: str):
        return ti_field_getattr(self.field, index, attribute)

    @ti.kernel
    def randomise(self):
        for i, j in ti.ndrange(self.shape, self.shape):
            # TODO: if DataType = ...
            state = {k: v[0] + (v[1] - v[0]) * ti.random(ti.f32)
                     for k, v in self.dict.items()}
            self.field[i,j] = self.struct(**state)
        # for i in ti.ndrange(**self.shape):
        #     state = ti.random()
        #     self.field[i] = self.struct()

    def set_args(self, index: tuple, *state: Any):
        self.field[index] = self.struct(*state)

    def set_kwargs(self, index: tuple, **state: Any):
        for k, v in self.dict.items():
            if k not in state:
                state[k] = ti_field_getattr(self.field, index, k)
        self.field[index] = self.struct(**state)

    def set_list(self, index: tuple, state: list):
        self.field[index] = self.struct(*state)

    def set_all_ndarray(self, state: ti.types.ndarray(dtype=ti.f32, ndim=3)):
        '''
        Should be @ti.kernel, but can't do `*state[i,j]`

        Args:
            state: np.array((shape, shape, state), dtype=np.float32)
        '''
        for i, j in ti.ndrange(state.shape[0], state.shape[0]):
            self.field[i,j] = self.struct(*state[i,j])

    def set_all_list(self, state: list):
        '''
        Flat list of state for each shape pair

        Args:
            state = [i0j0r0-i0j0rN, i0j1r0-i0j1rN, i1j1r0-i1jrN, ...]
        '''
        assert len(state) == self.len_ndarr_all, f"len(state) != len_ndarr_all ({len(state)} != {self.len_ndarr_all})"
        l = self.len
        for i, j in ti.ndrange(self.shape, self.shape):
            state = [state[i*l+j+r] for r in range(l)]
            self.field[i,j] = self.struct(*state)

    def set_shape_ndarray(self, i: ti.i32, state: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        '''
        Args:
            state: np.array((shape, state), dtype=np.float32)
        '''
        for j in range(self.shape):
            self.field[i,j] = self.struct(*state[j])

    def set_shape_list(self, i: ti.i32, state: list):
        '''
        Args:
            state = [i0r0-i0rN, i1r0-i1rN, ...]
        '''
        assert len(state) == self.len_ndarr_shape, f"len(state) != len_ndarr_shape ({len(state)} != {self.len_ndarr_shape})"
        l = self.len
        for j in range(self.shape):
            state = [state[j*l+r] for r in range(l)]
            self.field[i,j] = self.struct(*state)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if isinstance(args[0], tuple) and isinstance(args[1], str):
            return self.get(*args)
