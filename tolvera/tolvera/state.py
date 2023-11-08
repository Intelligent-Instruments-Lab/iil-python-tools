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
TODO: OSCMap getters
TODO: tidy up `osc_receive_randomise`, move into iipyper.map.py?
TODO: state analysers -> OSC senders
TODO: IML: add default mapping?
TODO: Sardine: pattern utils?
'''

from typing import Any
import taichi as ti
from .utils import Options
from taichi._lib.core.taichi_python import DataType
import numpy as np

@ti.data_oriented
class State:
    def __init__(self, 
                 options: Options,
                 state: dict[str, tuple[ti.f32, ti.f32]], # tuple[DataType, Any, Any]
                 shape: int,# | tuple[int], 
                 osc: bool = False,
                 name: str = 'state',
                 randomise: bool = True):
        self.o = options
        self.dict = state
        self.shape = shape
        # self.struct = ti.types.struct(**{k: v[0] for k,v in self.dict.items()})
        # self.field = self.struct.field(shape=self.shape)
        self.struct = ti.types.struct(**{k: ti.f32 for k,v in self.dict.items()})
        self.field = self.struct.field(shape=(self.shape, self.shape))
        self.len_state_idx = len(state)
        self.len_state_row = self.len_state_idx * self.shape#[0]
        self.len_state_col = self.len_state_idx * self.shape#[1]
        self.len_state_all = self.len_state_row * self.shape#[1] self.len_state_col
        self.len_attr_row = self.shape#[0]
        self.len_attr_col = self.shape#[1]
        self.len_attr_all = self.shape * self.shape
        self.osc = osc
        self.name = name
        self.init(randomise)

    def init(self, randomise: bool = False):
        if randomise: self.randomise()
        if self.osc:  self.add_to_osc_map()

    def get(self, index: tuple, attribute: str):
        return self.field[index][attribute]

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

    def set_state_idx_from_args(self, index: tuple, *state: Any):
        self.field[index] = self.struct(*state)

    def set_state_idx_from_kwargs(self, index: tuple, **state: Any):
        for k, v in self.dict.items():
            if k not in state:
                state[k] = self.field[index][k]
        self.field[index] = self.struct(**state)

    def set_state_idx_from_list(self, index: tuple, state: list):
        self.set_state_idx_from_args(index, *state)

    def set_state_row_from_list(self, i: int|tuple[int], state: list):
        '''
        Args:
            state = [i0r0-i0rN, i1r0-i1rN, ...]
        '''
        # assert len(state) == self.len_state_row, f"len(state) != len_ndarr_shape ({len(state)} != {self.len_state_row})"
        i = i[0] if isinstance(i, tuple) else i
        idx_len = self.len_state_idx
        for j in range(self.shape):
            s = [state[j*idx_len+r] for r in range(idx_len)]
            struct = self.struct(*s)
            self.field[i,j] = self.struct(*s)
    
    def set_state_col_from_list(self, j: int|tuple[int], state: list):
        '''
        Args:
            state = [i0r0-i0rN, i1r0-i1rN, ...]
        '''
        # assert len(state) == self.len_ndarr_shape, f"len(state) != len_ndarr_shape ({len(state)} != {self.len_ndarr_shape})"
        j = j[0] if isinstance(j, tuple) else j
        idx_len = self.len_state_idx
        for i in range(self.shape):
            s = [state[i*idx_len+r] for r in range(idx_len)]
            self.field[i,j] = self.struct(*s)

    def set_state_all_from_list(self, state: list):
        '''
        Flat list of state for each shape pair

        Args:
            state = [i0j0r0-i0j0rN, i0j1r0-i0j1rN, i1j1r0-i1jrN, ...]
        '''
        # assert len(state) == self.len_ndarr_all, f"len(state) != len_ndarr_all ({len(state)} != {self.len_ndarr_all})"
        idx_len = self.len_state_idx
        for i, j in ti.ndrange(self.shape, self.shape):
            s = [state[i*idx_len+j+r] for r in range(idx_len)]
            self.field[i,j] = self.struct(*s)

    def set_state_row_from_ndarray(self, i: ti.i32, state: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        '''
        Args:
            state: np.array((shape, state), dtype=np.float32)
        '''
        for j in range(self.shape):
            self.field[i,j] = self.struct(*state[j])
    
    def set_state_col_from_ndarray(self, j: ti.i32, state: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        '''
        Args:
            state: np.array((shape, state), dtype=np.float32)
        '''
        for i in range(self.shape):
            self.field[i,j] = self.struct(*state[i])

    def set_state_all_from_ndarray(self, state: ti.types.ndarray(dtype=ti.f32, ndim=3)):
        '''
        Should be @ti.kernel, but can't do `*state[i,j]`

        Args:
            state: np.array((shape, shape, state), dtype=np.float32)
        '''
        for i, j in ti.ndrange(state.shape[0], state.shape[0]):
            self.field[i,j] = self.struct(*state[i,j])
    
    def set_dim_from_ndarray(self, dim: ti.i32, state: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        raise NotImplementedError("set_dim_from_ndarray() not implemented")
    
    def set_dim_from_list(self, dim: ti.i32, state: list):
        raise NotImplementedError("set_dim_from_list() not implemented")
    
    def set_attr_idx(self, index: tuple, attr: str, value: Any):
        print('set_attr_idx',index, attr, value)
        value = value[0] if isinstance(value, list) else value
        self.field[index][attr] = value
    
    def set_attr_row(self, i: int|tuple[int], attr: str, values: list):
        i = i[0] if isinstance(i, tuple) else i
        for j in range(self.shape):
            self.field[i,j][attr] = values[i]
        
    def set_attr_col(self, j: int|tuple[int], attr: str, values: list):
        j = j[0] if isinstance(j, tuple) else j
        for i in range(self.shape):
            self.field[i,j][attr] = values[j]
    
    def set_attr_all(self, attr: str, values: list):
        for i, j in ti.ndrange(self.shape, self.shape):
            self.field[i,j][attr] = values[i*self.shape+j]
    
    def osc_receive_randomise(self):
        def randomise():
            self.randomise()
        return randomise

    def add_to_osc_map(self):
        name = f"{self.o.name_clean}_{self.name}"
        # Add randomise to OSCMap
        kwargs = {'count': 1, 'name': name+'_randomise'}
        self.o.osc_map.receive_args(**kwargs)(self.osc_receive_randomise())
        # Add state setters to OSCMap
        f = self.o.osc_map.receive_list_with_idx
        f(name+'_idx', self.set_state_idx_from_list, 2, getattr(self,'len_state_idx'))
        f(name+'_row', self.set_state_row_from_list, 1, getattr(self,'len_state_row'))
        f(name+'_col', self.set_state_col_from_list, 1, getattr(self,'len_state_col'))
        f(name+'_all', self.set_state_all_from_list, 0, getattr(self,'len_state_all'))
        # Add state attribute setters to OSCMap
        for k,v in self.dict.items():
            f(name+'_'+k+'_idx', self.set_attr_idx, 2, 1, k)
            f(name+'_'+k+'_row', self.set_attr_row, 1, getattr(self,'len_attr_row'), k)
            f(name+'_'+k+'_col', self.set_attr_col, 1, getattr(self,'len_attr_col'), k)
            f(name+'_'+k+'_all', self.set_attr_all, 0, getattr(self,'len_attr_all'), k)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if isinstance(args[0], tuple) and isinstance(args[1], str):
            return self.get(*args)
