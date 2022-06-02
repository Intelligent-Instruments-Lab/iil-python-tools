from .types import *
from .nnsearch import NNSearch
from . import feature
from . import interpolate

class IML:

    def __init__(self, feature_size:int, 
            feat=feature.Identity, interp=interpolate.Softmax):
        self.targets: Dict[TargetID, Target] = {}
        self.sources: Dict[TargetID, Source] = {}

        # Feature converts sources to feature vectors
        self.embed = feat(feature_size)

        # NNSearch converts feature to target IDs and scores
        self.neighbors = NNSearch(self.embed.size)

        # Interpolate combines a set of targets and scores
        self.interpolate = interp()

    # add a data point
    def add(self, source: Source, target: Target) -> None:
        feature = self.embed(source)
        target_id = self.neighbors.add(feature)
        # track the mapping from target IDs back to targets
        self.targets[target_id] = target
        self.sources[target_id] = source

    # convert a source to a target using embed, predict, interpolate
    def map(self, source: Source, k: Optional[int] = 5) -> Target:
        feature = self.embed(source)
        target_ids, scores = self.neighbors(feature)
        b = [i>=0 for i in target_ids] # case where there are fewer than k neighbors
        targets = [self.targets[i] for i in target_ids[b]]
        scores = scores[b]
        # print(targets, scores)
        return self.interpolate(targets, scores)


