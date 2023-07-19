from .types import *
from .nnsearch import NNSearch
from . import feature
from . import interpolate

class IML:

    def __init__(self, feature_size:Optional[int]=None, 
            embed=None, interp=None):
        
        # Feature converts sources to feature vectors
        if embed is None:
            embed = feature.Identity(feature_size)
        else:
            embed = getattr(feature, embed.capitalize())(feature_size)
        self.embed = embed

         # Interpolate combines a set of targets and scores
        if interp is None:
            self.interpolate = interpolate.Softmax()
        else:
            self.set_interp(interp)

        self.reset()
    
    def set_interp(self, name):
        self.interpolate = getattr(interpolate, name.capitalize())()

    def reset(self, keep_near:Optional[Source]=None, k:int=5):
        """delete all data
        Args:
            keep_near: don't remove the neighbors of this source
            k: number of neighbors for above
        """
        print('reset')
        if keep_near is not None and len(self.pairs)>0:
            if len(keep_near)!=len(self.pairs[0][0]):
                print('ERROR: iml: keep_near should be an input vector')
                keep_near = None
            else:
                print('searching neighbors for keep_near')
                srcs, tgts, _ = self.search(keep_near, k=k)

        self.pairs: Dict[TargetID, Tuple[Source, Target]] = {}
        # NNSearch converts feature to target IDs and scores
        self.neighbors = NNSearch(self.embed.size)

        if keep_near is not None:
            print(f'restoring {len(srcs)} neighbors')
            for s,t in zip(srcs,tgts):
                self.add(s,t)

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

        # TODO: text-mode visualize scores
        s = ' '*len(self.pairs)


        return sources, targets, scores

    def map(self, source: Source, k: int = 5, **kw) -> Target:
        """convert a source to a target using search + interpolate

        Args:
            source: input
            k: max neighbors
            **kw: additional arguments are passed to interpolate
        Returns:
            output instance
        """
        # print(f'map {source=}')
        _, targets, scores = self.search(source, k)
        result = self.interpolate(targets, scores, **kw)

        return result

    def save(self, path):
        """serialize: store dataset, config"""
        # TODO

    @classmethod
    def load(cls, path):
        """deserialize: construct a new IML from config, load dataset"""
        # TODO



