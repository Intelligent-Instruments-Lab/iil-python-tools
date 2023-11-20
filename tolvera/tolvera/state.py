'''
TODO: save/load? serialise/deserialise
TODO: OSCMap getters
    state analysers -> OSC senders
    iipyper osc returns?
TODO: tidy up `osc_receive_randomise`, move into iipyper.map.py?
TODO: IML: add default mapping?
TODO: Sardine: pattern utils?
TODO: @ti.func struct methods - can these be monkey patched?
    if not add to constructor as a dict
    use case would be Particles.Particle
'''

from typing import Any
from taichi._lib.core.taichi_python import DataType
import taichi as ti
import numpy as np

from .npndarray_dict import NpNdarrayDict, np_vec2, np_vec3, np_vec4

@ti.data_oriented
class State:
    '''
    Args
        tolvera: tolvera instance
        state: a dictionary of attributes in `'attr': (min, max)` format
        shape: the shape of the state field (currently only `int->(int,int)` supported)
        osc: one/both of `('get', 'set')` to create OSC getters and/or setters
        name: becomes the OSC path prefix for the state
        randomise: randomise the state on initialisation
    '''
    def __init__(self, 
                 tolvera,
                 state: dict[str, tuple[DataType, Any, Any]],
                 shape: int|tuple[int],
                 osc: tuple = None,
                 name: str = 'state',
                 randomise: bool = True):
        self.tv = tolvera
        self.dict = state
        self.shape = (shape,) if type(shape) is int else shape
        self.struct = ti.types.struct(**{k: v[0] for k,v in self.dict.items()})
        self.field = self.struct.field(shape=self.shape)
        self.osc = True if osc is not None else False
        self.osc_get = 'get' in osc if osc is not None else False
        self.osc_set = 'set' in osc if osc is not None else False
        self.name = name
        self.init(randomise)

    def init(self, randomise: bool = False):
        self.init_nddict()
        if randomise:
            self.nddict_randomise()
        if self.tv.osc is not False and self.osc:
            self.osc = self.tv.osc
            self.add_to_osc_map()

    def init_nddict(self):
        self.types = {
            ti.i32: np.int32,
            ti.f32: np.float32,
            ti.math.vec2: np_vec2,
            ti.math.vec3: np_vec3,
            ti.math.vec4: np_vec4,
        }

        nddict = {}
        for k, v in self.dict.items():
            titype, min_val, max_val = v
            nptype = self.types.get(titype)
            if nptype is None:
                raise NotImplementedError(f"no nptype for {titype}")
            nddict[k] = (nptype, min_val, max_val)
        
        self.nddict = NpNdarrayDict(nddict, self.shape)

    def from_nddict(self):
        self.field.from_numpy(self.nddict.get_data())

    def to_nddict(self):
        self.nddict.set_data(self.field.to_numpy())

    def nddict_randomise(self):
        self.nddict.randomise()
        self.from_nddict()

    def randomise(self):
        f_np = self.field.to_numpy()
        shape = self.shape
        for k, v in self.dict.items():
            datatype, dmin, dmax = v
            match datatype:
                case ti.i32:
                    f_np[k] = np.random.randint(dmin, dmax, size=shape).astype(np.int32)
                case ti.f32:
                    f_np[k] = np.random.uniform(dmin, dmax, size=shape).astype(np.float32)
                case ti.math.vec2:
                    f_np[k] = np.random.uniform(dmin, dmax, size=shape+(2,)).astype(np.float32)
                case ti.math.vec3:
                    f_np[k] = np.random.uniform(dmin, dmax, size=shape+(3,)).astype(np.float32)
                case ti.math.vec4:
                    f_np[k] = np.random.uniform(dmin, dmax, size=shape+(4,)).astype(np.float32)
                case _:
                    raise NotImplementedError(f"randomise() not implemented for {datatype}")
        self.field.from_numpy(f_np)

    @ti.kernel
    def ti_randomise(self):
        """
        TODO: every `I`, need to generate a random self.struct
            for this we need to retrieve datatype, dmin, dmax for each attribute as above
            dont know how to do this in Taichi scope
        """
        for I in ti.grouped(self.field):
            self.field[I] = self.struct()

    def osc_getter(self, i: int, j: int, attribute: str):
        ret = self.get((i,j), attribute)
        if ret is not None:
            route = self.osc.map.pascal_to_path(self.getter_name)#+'/'+attribute
            self.osc.host.return_to_sender_by_name((route, attribute,ret), self.osc.client_name)
        return ret

    def add_to_osc_map(self):
        if self.osc_set:
            self.setter_name = f"{self.tv.name_clean}_set_{self.name}"
            self.add_osc_setters(self.setter_name)
        if self.osc_get:
            self.getter_name = f"{self.tv.name_clean}_get_{self.name}"
            self.add_osc_getters(self.getter_name)

    def add_osc_setters(self, name):
        # randomise
        self.osc.map.receive_args_inline(name+'_randomise', self._randomise)
        # state
        f = self.osc.map.receive_list_with_idx
        f(f"{name}_idx", self.set_state_idx_from_list, 2, getattr(self,'len_state_idx'))
        f(f"{name}_row", self.set_state_row_from_list, 1, getattr(self,'len_state_row'))
        f(f"{name}_col", self.set_state_col_from_list, 1, getattr(self,'len_state_col'))
        f(f"{name}_all", self.set_state_all_from_list, 0, getattr(self,'len_state_all'))
        # state attributes
        for k,v in self.dict.items():
            f(f"{name}_{k}_idx", self.set_attr_idx, 2, 1, k)
            f(f"{name}_{k}_row", self.set_attr_row, 1, getattr(self,'len_attr_row'), k)
            f(f"{name}_{k}_col", self.set_attr_col, 1, getattr(self,'len_attr_col'), k)
            f(f"{name}_{k}_all", self.set_attr_all, 0, getattr(self,'len_attr_all'), k)

    def add_osc_getters(self, name):
        for k,v in self.dict.items():
            ranges = (int(v[0]), int(v[0]), int(v[1]))
            kwargs = {'i': ranges, 'j': ranges, 'attr': (k,k,k)}
            self.osc.map.receive_args_inline(f"{name}", self.osc_getter, **kwargs)

    '''
    def add_osc_streams(self, name):
        add in broadcast mode
        pass
    '''

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
