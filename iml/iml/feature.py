from .types import *

import numpy as np

class Feature:
    def __init__(self, size):
        self.size = size

    def __call__(self, source: Source) -> Feature:
        raise NotImplementedError

class Identity(Feature):
    """just check size and convert to numpy array"""
    def __call__(self, source):
        source = np.array(source)
        assert source.shape[-1] == self.size
        return source
