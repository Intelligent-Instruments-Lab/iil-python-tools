import numpy as np
from typing import Any, Optional, List, Tuple, Dict, Union, Callable, Generator
from typing import NamedTuple
from numpy.typing import ArrayLike

from collections import namedtuple

Input = Any # thing that goes into a mapping
Output = Any # thing that comes out of a mapping
Feature = ArrayLike # Inputs are mapped to Features
Scores = ArrayLike # Scores describe distance between inputs in feature space
PairID = int # PairIDs associate Inputs (via Features) with Outputs
PairIDs = ArrayLike

class IDFeaturePair(NamedTuple):
    id:PairID
    feature:Feature
class IOPair(NamedTuple):
    input:Input
    output:Output
class SearchResult(NamedTuple):
    inputs:List[Input]
    outputs:List[Output]
    ids:PairIDs
    scores:Scores
# IDFeaturePair = namedtuple('IDFeaturePair', ['id', 'feature'])
# IOPair = namedtuple('IOPair', ['input', 'output'])
# SearchResult = namedtuple('SearchResult', ['inputs', 'outputs', 'ids', 'scores'])

def _np_coerce(x):
    if hasattr(x, 'numpy'):
        return x.numpy()
    elif hasattr(x, '__len__') and hasattr(x[0], 'numpy'):
        return np.stack(tuple(item.numpy() for item in x))
    else:
        return np.array(x)

def np_coerce(*a):
    return (_np_coerce(x) for x in a)