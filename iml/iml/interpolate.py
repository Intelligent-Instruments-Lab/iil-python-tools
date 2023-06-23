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
    def __init__(self):
        super().__init__()

    def __call__(self, targets, scores, temp=1):
        targets, scores = np_coerce(targets, scores)
        # print(targets.shape, scores.shape)

        if temp==0:
            result = targets[np.argmax(scores)]
        else:
            centered = scores - np.mean(scores) # for numerical precision
            logits = np.maximum(-centered/temp, -20)
            # print(f'{logits=}')
            if np.max(np.abs(logits)) > 80:
                result = targets[np.argmax(scores)]
            else:
                weights = np.exp(logits)
                # print(f'{weights=}')
                weights /= weights.sum()
                # print(f'{weights=}')
                result = (np.moveaxis(targets,0,-1)*weights).sum(-1)
        # print(f'{result=}')
        return result

class Smooth(Interpolate):
    """set the furthest neighbor weight to zero"""
    def __init__(self):
        super().__init__()

    def __call__(self, targets, scores, temp=1):
        targets, scores = np_coerce(targets, scores)

        # print(f'{scores=}')
        scores = scores + 1e-9
        scaled = scores / np.max(scores)
        # print(f'{scaled=}')
        weights = (1/scaled - 1)**temp  # [index]
        # print(f'{weights=}')
        weights /= weights.sum()
        result = (np.moveaxis(targets,0,-1)*weights).sum(-1)

        return result
    
class Ripple(Interpolate):
    """"""
    def __init__(self):
        super().__init__()

    def __call__(self, targets, scores, ripple=1):
        targets, scores = np_coerce(targets, scores)

        scores = scores + 1e-9
        scores = scores - np.min(scores)
        scores = scores / np.max(scores)

        scores = 1 + np.cos(np.pi * scores * ripple)

        weights = scores / scores.sum()

        result = (np.moveaxis(targets,0,-1)*weights).sum(-1)

        return result