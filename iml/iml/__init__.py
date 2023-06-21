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
        self.pairs: Dict[TargetID, Tuple[Source, Target]] = {}
        # NNSearch converts feature to target IDs and scores
        self.neighbors = NNSearch(self.embed.size)

    def add(self, source: Source, target: Target) -> TargetID:
        """add a data point
        Args:
            source: input item
            target: output item
        Returns:
            target_id: id of the new data point (you may not need this)
        """
        print(f'add {source=}, {target=}')
        feature = self.embed(source)
        target_id = self.neighbors.add(feature)
        # track the mapping from target IDs back to targets
        self.pairs[target_id] = (source, target)
        return target_id

    def search(self, source: Source, k: int = 5):# -> Tuple[List[Source], List[Target]]:
        """find k-nearest neighbors
        Args:
            source: input item
            k: max number of neighbors
        Returns:
            sources: neighboring inputs
            targets: corresponding outputs
            scores: dissimilarity scores
        """
        feature = self.embed(source)
        target_ids, scores = self.neighbors(feature, k=k)
        # handle case where there are fewer than k neighbors
        sources, targets = zip(*(self.pairs[i] for i in target_ids))
        return sources, targets, scores

    def map(self, source: Source, k: int = 5) -> Target:
        """convert a source to a target using search + interpolate"""
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



