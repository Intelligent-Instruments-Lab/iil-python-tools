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
            k:int=10,
            verbose=False):
        """
        Args:
            feature_size: dimension of feature vectors
            embed: instance or name of Feature subclass (defaults to Identity)
            interp: instance or name of Interpolate subclass (defaults to Smooth)
            index: instance of Index (defaults to IndexBrute)
            k: default k-nearest neighbors (can be overridden later)
        """
        self.verbose = verbose
        # Feature converts Inputs to Features
        if emb is None:
            emb = embed.Identity(feature_size)
        elif isinstance(emb, str):
            emb = getattr(embed, emb)(feature_size)
        elif isinstance(emb, embed.Embedding):
            pass
        else:
            raise ValueError

         # Interpolate combines a set of Outputs according to their Scores
        if interp is None:
            interp = interpolate.Smooth()
        elif isinstance(interp, str):
            interp = getattr(interpolate, interp)()
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
        res = None
        if keep_near is not None and len(self.pairs)>0:
            if len(keep_near)!=len(self.pairs[0][0]):
                print('ERROR: iml: keep_near should be an Input vector')
                keep_near = None
            else:
                print('searching neighbors for keep_near')
                res = self.search(keep_near, k=k)

        self.pairs: Dict[PairID, IOPair] = {}
        # NNSearch converts feature to output IDs and scores
        self.neighbors.reset()

        if res is not None:
            print(f'restoring {len(res.ids)} neighbors')
            for id,inp,out in zip(res.ids, res.inputs, res.outputs):
                self.add(inp,out,id=id)

    def add(self, 
            input: Input, 
            output: Output, 
            id: Optional[PairID]=None
            ) -> PairID:
        """Add a data point to the mapping.
        Args:
            input: Input item
            output: Output item
            id: PairID to use; if an existing id, replace the point
        Returns:
            id: id of the new data point if you need to reference it later
        """
        if self.verbose: print(f'add {input=}, {output=}')
        feature = self.embed(input)
        id = self.neighbors.add(feature, id)
        # track the mapping from output IDs back to outputs
        self.pairs[id] = IOPair(input, output)
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

    def remove_near(self, input:Input, k:int=None):
        """Remove from mapping by proximity to Input
        """
        feature = self.embed(input)
        self.neighbors.remove_near(feature)

    def search(self, input:Input, k:int=None) -> SearchResult:
        """find k-nearest neighbors
        Args:
            input: input item
            k: max number of neighbors
        Returns:
            inputs: neighboring Inputs
            outputs: corresponding Outputs
            ids: ids of Input/Output pairs
            scores: dissimilarity Scores
        """
        feature = self.embed(input)
        ids, scores = self.neighbors(feature, k=k)
        # handle case where there are fewer than k neighbors
        if not len(ids):
            raise RuntimeError('no points in mapping. add some!')
        inputs, outputs = zip(*(self.pairs[i] for i in ids))

        # TODO: text-mode visualize scores
        # s = ' '*len(self.pairs)

        return SearchResult(inputs, outputs, ids, scores)

    def map(self, input:Input, k:int=None, **kw) -> Output:
        """convert an Input to an Output using search + interpolate

        Args:
            input: input
            k: max neighbors
            **kw: additional arguments are passed to interpolate
        Returns:
            output instance
        """
        # print(f'map {input=}')
        _, outputs, _, scores = self.search(input, k)
        result = self.interpolate(outputs, scores, **kw)

        return result
    
    def save_state(self):
        """return dataset from this IML object.
        Returns:
            state: data in this IML object
        """
        return {
            'pairs': self.pairs
        }
    
    def load_state(self, state):
        """load dataset into this IML object.
        Args:
            state: data as obtained from `save_state`
        """
        for id,pair in state['pairs'].items():
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

