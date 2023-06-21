import numpy as np
from typing import Any, Optional, List, Tuple, Dict
from numpy.typing import ArrayLike

Source = Any
Target = Any
Feature = ArrayLike
Scores = ArrayLike
TargetID = int
TargetIDs = ArrayLike[TargetID]

def _np_coerce(x):
    if hasattr(x, 'numpy'):
        return x.numpy()
    elif hasattr(x, '__len__') and hasattr(x[0], 'numpy'):
        return np.stack(tuple(item.numpy() for item in x))
    else:
        return np.array(x)

def np_coerce(*a):
    return (_np_coerce(x) for x in a)