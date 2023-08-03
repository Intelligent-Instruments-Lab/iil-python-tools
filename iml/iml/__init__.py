from .types import *
from . import nnsearch
from . import feature
from . import interpolate

class IML:
    def __init__(self, 
            feature_size:Optional[int]=None, 
            embed:Union[str,feature.Feature]=None, 
            interp:Union[str,interpolate.Interpolate]=None,
            index:nnsearch.Index=None,
            k:int=10,
            verbose=False):
        """
        Args:
            feature_size: dimension of feature vectors
            embed: instance or name of Feature subclass (defaults to Identity)
            interp: instance or name of Interpolate subclass (defaults to Smooth)
            index: instance of Index (defaults to IndexBrute)
            k: default k-nearest neighbors (can be overridden later)
            verbose: verbose debug printing
        """
        
        # Feature converts Inputs to Features
        if embed is None:
            self.embed = feature.Identity(feature_size)
        elif isinstance(embed, str):
            self.embed = getattr(feature, embed)(feature_size)
        elif isinstance(embed, feature.Feature):
            self.embed = embed
        else:
            raise ValueError

         # Interpolate combines a set of Outputs according to their Scores
        if interp is None:
            self.interpolate = interpolate.Smooth()
        elif isinstance(interp, str):
            self.interpolate = getattr(interpolate, interp.capitalize())()
        elif isinstance(interp, interpolate.Interpolate):
            self.interpolate = interp
        else:
            raise ValueError

        # an index determines the distance metric and efficiency
        if index is None:
            index = nnsearch.IndexBrute(self.embed.size)
        self.neighbors = nnsearch.NNSearch(index, k=k)

        self.verbose = verbose

        self.reset()

    def reset(self, keep_near:Input=None, k:int=None):
        """delete all data
        Args:
            keep_near: don't remove the neighbors of this input
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

        self.pairs: Dict[PairID, IOPair] = {}
        # NNSearch converts feature to target IDs and scores
        self.neighbors.reset()

        if keep_near is not None:
            print(f'restoring {len(srcs)} neighbors')
            for s,t in zip(srcs,tgts):
                self.add(s,t)

    def add(self, source: Input, target: Output) -> PairID:
        """Add a data point to the mapping
        Args:
            source: Input item
            target: Output item
        Returns:
            target_id: id of the new data point (you may not need this)
        """
        if self.verbose: print(f'add {source=}, {target=}')
        feature = self.embed(source)
        target_id = self.neighbors.add(feature)
        # track the mapping from target IDs back to targets
        self.pairs[target_id] = IOPair(source, target)
        return target_id
    
    def get(self, id:PairID) -> IOPair:
        """look up an Input/Output pair by ID"""
        try:
            return self.pairs[id]
        except Exception:
            print("NNSearch: WARNING: can't `get` ID which doesn't exist or has been removed")

    def remove(self, ids:Union[PairID, PairIDs]):
        """Remove from mapping by ID(s)
        """
        # iterable of ids case:
        if hasattr(ids, '__len__'):
            for id in ids:
                self.remove(id)
        # single id case:
        else:
            try:
                del self.pairs[ids]
            except Exception:
                print(f"IML: WARNING: can't `remove` ID {ids} which doesn't exist or has already been removed")
            self.neighbors.remove(ids)

    def remove_near(self, source:Input, k:int=None):
        """Remove from mapping by proximity to Input
        """
        feature = self.embed(source)
        self.neighbors.remove_near(feature)

    def search(self, source:Input, k:int=None) -> SearchResult:
        """find k-nearest neighbors
        Args:
            source: input item
            k: max number of neighbors
        Returns:
            sources: neighboring Inputs
            targets: corresponding Outputs
            ids: ids of Input/Output pairs
            scores: dissimilarity Scores
        """
        feature = self.embed(source)
        ids, scores = self.neighbors(feature, k=k)
        # handle case where there are fewer than k neighbors
        sources, targets = zip(*(self.pairs[i] for i in ids))

        # TODO: text-mode visualize scores
        # s = ' '*len(self.pairs)

        return SearchResult(sources, targets, ids, scores)

    def map(self, source:Input, k:int=None, **kw) -> Output:
        """convert an Input to an Output using search + interpolate

        Args:
            source: input
            k: max neighbors
            **kw: additional arguments are passed to interpolate
        Returns:
            output instance
        """
        # print(f'map {source=}')
        _, targets, _, scores = self.search(source, k)
        result = self.interpolate(targets, scores, **kw)

        return result

    def save(self, path):
        """serialize: store dataset, config"""
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        """deserialize: construct a new IML from config, load dataset"""
        raise NotImplementedError



