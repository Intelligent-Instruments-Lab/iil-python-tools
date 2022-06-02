import numpy as np

from .types import *

class Interpolate:
    def __init__(self):
        pass

    def __call__(self, targets: List[Target], scores: Scores) -> Target:
        raise NotImplementedError


class Softmax(Interpolate):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha

    def __call__(self, targets, scores):
        # target_t = type(targets[0])
        targets = np.array(targets) # index, ...
        centered = scores - np.mean(scores) # for numerical precision
        logits = np.maximum(-scores*self.alpha, -20)  # index
        weights = np.exp(logits)
        weights /= weights.sum()
        result = (np.moveaxis(targets,0,-1)*weights).sum(-1)
        # assumes the target type can be constructed from numpy array
        # return target_t(result)
        return result
