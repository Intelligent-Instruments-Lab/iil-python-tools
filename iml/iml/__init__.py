from .types import *
from . import nnsearch
from . import embed
from . import interpolate
from . import serialize

# TODO: state serialization
# TODO: serialize defaults where possible

class IML(serialize.JSONSerializable):

    def __init__(self, 
            feature_size:Optional[int]=None, 
            emb:Union[str,embed.Embedding]=None, 
            interp:Union[str,interpolate.Interpolate]=None,
            index:nnsearch.Index=None,
            k:int=10):
        """
        Args:
            feature_size: dimension of feature vectors
            embed: instance or name of Feature subclass (defaults to Identity)
            interp: instance or name of Interpolate subclass (defaults to Smooth)
            index: instance of Index (defaults to IndexBrute)
            k: default k-nearest neighbors (can be overridden later)
        """
        
        # Feature converts Inputs to Features
        if emb is None:
            emb = embed.Identity(feature_size)
        elif isinstance(emb, str):
            emb = getattr(emb, emb.capitalize())(feature_size)
        elif isinstance(emb, embed.Embedding):
            pass
        else:
            raise ValueError

         # Interpolate combines a set of Outputs according to their Scores
        if interp is None:
            interp = interpolate.Smooth()
        elif isinstance(interp, str):
            interp = getattr(interpolate, interp.capitalize())()
        elif isinstance(interp, interpolate.Interpolate):
            pass
        else:
            raise ValueError

        # Index determines the distance metric and efficiency
        if index is None:
            index = nnsearch.IndexBrute(emb.size)
        
        super().__init__(
            feature_size=feature_size, 
            emb=emb, interp=interp, index=index,
            k=k)

        self.interpolate = interp
        self.embed = emb
        self.neighbors = nnsearch.NNSearch(index, k=k)
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
                print('ERROR: iml: keep_near should be an Input vector')
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

    def add(self, 
            source: Input, 
            target: Output, 
            id: Optional[PairID]=None
            ) -> PairID:
        """Add a data point to the mapping.
        Args:
            source: Input item
            target: Output item
            id: PairID to use; if an existing id, replace the point
        Returns:
            id: id of the new data point if you need to reference it later
        """
        print(f'add {source=}, {target=}')
        feature = self.embed(source)
        id = self.neighbors.add(feature, id)
        # track the mapping from target IDs back to targets
        self.pairs[id] = IOPair(source, target)
        return id
    
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
            inputs: neighboring Inputs
            outputs: corresponding Outputs
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
    
    def save_state(self):
        """return dataset from this IML object.
        Returns:
            state: data in this IML object
        """
        return self.pairs
    
    def load_state(self, state):
        """load dataset into this IML object.
        Args:
            state: data as obtained from `save_state`
        """
        for id,pair in state.items():
            self.add(*pair, id=PairID(id))        

    def save(self, path:str):
        """serialize the whole IML object to JSON
        Args:
            path: path to JSON file
        """
        serialize.save(path, self)

    @classmethod
    def load(cls, path):
        """deserialize a new IML object from JSON
        Args:
            path: path to JSON file
        Returns:
            new IML instance
        """
        inst = serialize.load(path)
        assert isinstance(inst, cls), type(inst)
        return inst


