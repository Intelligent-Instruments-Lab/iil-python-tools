'''
TODO: Save/load
'''

from typing import Any
import unicodedata
import numpy as np
import jsons
import base64
import time

import taichi as ti
from taichi.lang.field import ScalarField
from taichi._lib.core.taichi_python import DataType

def remove_accents(input: str):
    nfkd_form = unicodedata.normalize('NFKD', input)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def clean_name(name: str):
    return remove_accents(name).strip().lower()

# @ti.data_oriented
class CONSTS:
    '''
    Dict of CONSTS that can be used in Taichi scope
    '''
    def __init__(self, dict: dict[str, (DataType, Any)]):
        self.struct = ti.types.struct(**{k: v[0] for k, v in dict.items()})
        self.consts = self.struct(**{k: v[1] for k, v in dict.items()})
    def __getattr__(self, name):
        try:
            return self.consts[name]
        except:
            raise AttributeError(f"CONSTS has no attribute {name}")
    def __getitem__(self, name):
        try:
            return self.consts[name]
        except:
            raise AttributeError(f"CONSTS has no attribute {name}")

def ndarray_b64_serialize(ndarray):
    return {
        "@type": "ndarray",
        "dtype": str(ndarray.dtype),
        "shape": ndarray.shape,
        "b64": base64.b64encode(ndarray.tobytes()).decode('utf-8')
    }

def ndarray_b64_deserialize(serialized):
    return np.frombuffer(base64.b64decode(serialized["b64"]), dtype=np.dtype(serialized["dtype"])).reshape(serialized["shape"])

def ti_serialize(field):
    if isinstance(field, (ScalarField, ti.lang.struct.StructField, ti.lang.matrix.MatrixField, ti.lang.matrix.VectorNdarray, ti.lang._ndarray.ScalarNdarray)):
        ndarray = field.to_numpy()
        if isinstance(ndarray, dict): # For StructField where to_numpy() returns a dict
            serialized = jsons.dumps({k: ndarray_b64_serialize(v) for k, v in ndarray.items()})
        else: # For other fields
            serialized = jsons.dumps(ndarray_b64_serialize(ndarray))
        field.serialized = serialized
        return serialized
    else:
        raise TypeError(f"Unsupported field type for serialization: {type(field)}")

def ti_deserialize(field, json_str):
    if isinstance(field, (ScalarField, ti.lang.struct.StructField, ti.lang.matrix.MatrixField, ti.lang.matrix.VectorNdarray, ti.lang._ndarray.ScalarNdarray)):
        data = jsons.loads(json_str)
        if isinstance(field, ti.lang.struct.StructField): # For StructField
            field.from_numpy({k: ndarray_b64_deserialize(v) for k, v in data.items()})
        else: # For other fields
            field.from_numpy(ndarray_b64_deserialize(data))
        field.serialized = None
    else:
        raise TypeError(f"Unsupported field type for deserialization: {type(field)}")

def time_function(func, *args, **kwargs):
    """Time how long it takes to run a function and print the result
    """
    start = time.time()
    func(*args, **kwargs)
    end = time.time()
    print(f"[Tolvera.utils] {func.__name__}() ran in {end-start:.4f}s")
    return end-start