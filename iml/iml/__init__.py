from .types import *
from .nnsearch import NNSearch
from . import feature
from . import interpolate

class IML:

    def __init__(self, feature_size:Optional[int]=None, 
            embed=None, interp=None):
        
        if embed is None:
            embed=feature.Identity(feature_size)
        if interp is None:
            interp = interpolate.Softmax()
        # Feature converts sources to feature vectors
        self.embed = embed
         # Interpolate combines a set of targets and scores
        self.interpolate = interp

        self.reset()

    def reset(self):
        """delete all data"""
        print('reset')
        self.targets: Dict[TargetID, Target] = {}
        self.sources: Dict[TargetID, Source] = {}
        # NNSearch converts feature to target IDs and scores
        self.neighbors = NNSearch(self.embed.size)

    def add(self, source: Source, target: Target) -> None:
        """add a data point"""
        print(f'add {source=}, {target=}')
        feature = self.embed(source)
        target_id = self.neighbors.add(feature)
        # track the mapping from target IDs back to targets
        self.targets[target_id] = target
        self.sources[target_id] = source

    def search(self, source: Source, k: int = 5):# -> Tuple[List[Source], List[Target]]:
        feature = self.embed(source)
        target_ids, scores = self.neighbors(feature, k=k)
        # handle case where there are fewer than k neighbors
        b = [i>=0 for i in target_ids] 
        tid = target_ids[b]
        sources = [self.sources[i] for i in tid]
        targets = [self.targets[i] for i in tid]
        scores = scores[b]
        return sources, targets, scores

    def map(self, source: Source, k: int = 5) -> Target:
        """convert a source to a target using embed, neighbors, interpolate"""
        # # print(f'map {source=}')
        # feature = self.embed(source)
        # # print(f'{feature=}')
        # target_ids, scores = self.neighbors(feature, k=k)
        # # print(f'{target_ids=}')
        # b = [i>=0 for i in target_ids] # case where there are fewer than k neighbors
        # # print(f'{b=}')
        # targets = [self.targets[i] for i in target_ids[b]]
        # # print(f'{targets=}')
        # scores = scores[b]
        _, targets, scores = self.search(source, k)
        # print(f'{scores=}')
        result = self.interpolate(targets, scores)
        # print(f'{result=}')
        return result

    def save(self, path):
        """serialize: store dataset, config"""
        # TODO

    @classmethod
    def load(cls, path):
        """deserialize: construct a new IML from config, load dataset"""
        # TODO



