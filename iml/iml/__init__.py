from .types import *
from .nnsearch import NNSearch
from . import feature
from . import interpolate

class IML:

    def __init__(self, feature_size:int, 
            feat=feature.Identity, interp=interpolate.Softmax):
        # Feature converts sources to feature vectors
        self.embed = feat(feature_size)
         # Interpolate combines a set of targets and scores
        self.interpolate = interp()

        self.reset()

    def reset(self):
        """delete all data"""
        self.targets: Dict[TargetID, Target] = {}
        self.sources: Dict[TargetID, Source] = {}
        # NNSearch converts feature to target IDs and scores
        self.neighbors = NNSearch(self.embed.size)

    def add(self, source: Source, target: Target) -> None:
        """add a data point"""
        print(source)
        feature = self.embed(source)
        target_id = self.neighbors.add(feature)
        # track the mapping from target IDs back to targets
        self.targets[target_id] = target
        self.sources[target_id] = source

    def map(self, source: Source, k: Optional[int] = 5) -> Target:
        """convert a source to a target using embed, predict, interpolate"""
        feature = self.embed(source)
        target_ids, scores = self.neighbors(feature, k=k)
        b = [i>=0 for i in target_ids] # case where there are fewer than k neighbors
        targets = [self.targets[i] for i in target_ids[b]]
        scores = scores[b]
        # print(targets, scores)
        return self.interpolate(targets, scores)

    def save(self, path):
        """serialize: store dataset, config"""
        # TODO

    @classmethod
    def load(cls, path):
        """deserialize: construct a new IML from config, load dataset"""
        # TODO



