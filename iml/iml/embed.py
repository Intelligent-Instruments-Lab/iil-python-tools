from .types import *
from .serialize import JSONSerializable

import numpy as np

class Embedding(JSONSerializable):
    def __init__(self, **kw):
        super().__init__(**kw)

    def __call__(self, source: Input) -> Feature:
        raise NotImplementedError

class Identity(Embedding):
    """For fixed-length vector data: just check size and convert to numpy array"""
    def __init__(self, size):
        """size is both the Input and Feature size"""
        super().__init__(size=size)
        self.size = self.input_size = size

    def __call__(self, source):
        source, = np_coerce(source)
        if self.size is not None:
            assert source.shape[-1] == self.size, (source.shape, self.size)
        return source
    
class ProjectAndSort(Embedding):
    """
    for point cloud-like data.
    use with L2 distance to compute sliced optimal transport.

    if an Input is a 2D array [B x C],
    B being the batch dimension (order not meaningful)
    and C being the coordinate dimension (order meaningful)

    e.g.
    [
    [x0,y0,z0],
    [x1,y1,z1],
    [x2,y2,z2],
    [x3,y3,z3],
    ]
    would be a cloud of B=4 points in C=3 dimensional space

    This computes `n` pseudo-random projections down to 1-dimensional spaces,
    sorts along those lines,
    and then concatenates to make one feature vector.
    the L2 distance between feature vectors is the sliced OT distance between point clouds.
    """
    def __init__(self, input_size:Tuple[int,int]=None, n:int=16):
        """
        Args:
            input_size: input shape [B,C]; if None, lazy init on first __call__
            n: number of random projections.
        """
        super().__init__(input_size=input_size, n=n)
        assert len(input_size)==2, "ProjectAndSort expects fixed-size 2D array data"

        self.n = n
        if input_size is not None:
            self.init(input_size)
        else:
            self.input_size = None

    def init(self, input_size):
        self.input_size = tuple(input_size)

        self.size = input_size[0] * self.n

        proj = np.random.randn(input_size[1], self.n)
        proj = proj / np.linalg.norm(proj, axis=0, keepdims=True)
        self.proj = proj

    def __call__(self, source):
        source, = np_coerce(source)
        if self.input_size is None:
            # lazy init
            self.init(source.shape[-2:])
        else:
            assert source.shape[-2:] == self.input_size, (source.shape, self.input_size)

        # project coordinate dimension to n lines
        feat = source @ self.proj
        # sort along the lines
        feat = np.sort(feat, axis=-2)
        # flatten
        # feat = feat.T
        feat = feat.reshape((*feat.shape[:-2], -1))

        return feat / np.sqrt(self.size)
