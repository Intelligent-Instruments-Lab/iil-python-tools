import numpy as np

from .types import *

class Interpolate:
    """
    Interpolate combines a set of Targets weighted by similarity scores.
    """
    def __init__(self):
        pass

    def __call__(self, targets: List[Target], scores: Scores) -> Target:
        raise NotImplementedError


class Softmax(Interpolate):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha

    def __call__(self, targets, scores):
        targets, scores = np_coerce(targets, scores)
        # print(targets.shape, scores.shape)
        # target_t = type(targets[0])

        centered = scores - np.mean(scores) # for numerical precision
        logits = np.maximum(-centered*self.alpha, -20)  # index
        weights = np.exp(logits)
        weights /= weights.sum()
        result = (np.moveaxis(targets,0,-1)*weights).sum(-1)
        # assumes the target type can be constructed from numpy array
        # return target_t(result)
        return result

class Smooth(Interpolate):
    """set the furthest neighbor weight to zero"""
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha

    def __call__(self, targets, scores):
        targets, scores = np_coerce(targets, scores)

        # target_t = type(targets[0])
        scores = scores + 1e-9
        scaled = scores / np.max(scores)
        weights = (1/scaled - 1)**self.alpha  # [index]
        weights /= weights.sum()
        result = (np.moveaxis(targets,0,-1)*weights).sum(-1)
        # assumes the target type can be constructed from numpy array
        # return target_t(result)
        return result
