import numpy as np
from .types import *    
from .serialize import JSONSerializable

class Metric(JSONSerializable):
    def __call__(self, a, b):
        raise NotImplementedError

class sqL2(Metric):
    def __call__(self, a, b):
        return np.sum((a-b)**2, axis=-1)

class Index(JSONSerializable):
    """base Index class.
    currently no function besides typing, warning of unimplemented features
    """
    def add(self, feature:Feature, id:Optional[PairID]=None):
        raise NotImplementedError
    def remove(self, id:PairID):
        raise NotImplementedError
    def get(self, id:PairID):
        raise NotImplementedError
    def search(self, feature:Feature, k:int) -> Tuple[Scores, PairIDs]:
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError
    @property
    def ids(self):
        raise NotImplementedError

class IndexBrute(Index):
    """
    Optimized for simplicity and flexibility,
    may not scale to large datasets
    """
    def __init__(self, d:int=None, metric:Callable=None):
        """
        Args:
            d: optional, dimension of feature
            metric: distance metric, default to squared euclidean
        """
        if metric is None:
            metric = sqL2()
        
        super().__init__(d=d, metric=metric)
        self.d = d
        self.metric = metric

        self.reset()

    def add(self, feature:Feature, id:Optional[PairID]=None) -> PairID:
        """add a new feature, return its ID.
        Args:
            feature: the feature to add
            id: if not supplied, generate a new ID;
                otherwise, use the supplied id.
                supply an existing id to replace.
        """
        if id is None:
            id = max(self.data, default=-1) + 1
        self.data[id] = feature
        return id
    
    def remove(self, id:PairID):
        """remove a feature by ID"""
        del self.data[id]
    
    def get(self, id:PairID) -> Feature:
        """get a feature by ID"""
        return self.data[id]

    def search(self, feature:Feature, k:int=3) -> Tuple[PairIDs, Scores]:
        """get feature(s) and IDs by proximity"""
        if not len(self.data):
            return [], []
        dist_id = sorted((self.metric(feature, v),k) for k,v in self.data.items())
        scores, ids = zip(*dist_id[:k])
        return ids, scores
    
    def reset(self):
        self.data:Dict[PairID, Feature] = {}

    @property
    def ids(self):
        return self.data.keys()
    
try:
    import faiss
    from faiss import IndexFlatL2
    class IndexFast(Index):
        """
        Optimized for fast `search` on large vectors / datasets.
        Only L2 distance supported. 
        `remove` may be slow.

        This is currently a wrapper around faiss.FlatIndexL2 which provides stable ids when using `remove`
        In the future could support dot product and/or approximate search indices.
        """
        def __init__(self, d:int, metric:Callable=sqL2):
            """
            Args:
                d: dimension of feature
                metric: 
            """
            super().__init__(d=d, metric=metric)
            if metric==sqL2:
                self.index = IndexFlatL2(d)
            else:
                raise ValueError("""IndexFast supports only sqL2 metric""")
            self.metric = metric
            self.reset()

        @property
        def d(self):
            return self.index.d

        def add(self, feature:Feature, id:Optional[PairID]=None):
            """add a new feature, return its ID.
            Args:
                feature: the feature to add
                id: if not supplied, generate a new ID;
                    otherwise, use the supplied id.
                    supply an existing id to replace.
            """
            if id is None:
                # no id supplied case
                id = max(self.id_to_idx, default=-1) + 1
            elif id in self.id_to_idx:
                # existing id supplied case
                self.remove(id)
            feature = feature[None].astype(np.float32)
            self.index.add(feature)
            idx = self.index.ntotal - 1
            # map ID to faiss index
            self.id_to_idx[id] = idx
            self.idx_to_id[idx] = id
            return id

        def remove(self, id:PairID):
            """remove a feature by ID"""
            idx = self.id_to_idx[id]
            self.index.remove_ids(np.array(idx)[None])
            del self.id_to_idx[id]
            del self.idx_to_id[idx]
            # faiss shifts its internal index to stay dense
            self.id_to_idx = {
                k:(v-1 if v > idx else v) for k,v in self.id_to_idx.items()}
            self.idx_to_id = {
                (k-1 if k > idx else k):v for k,v in self.idx_to_id.items()}
        
        def get(self, id:PairID):
            """get a feature by ID"""
            idx = self.id_to_idx[id]
            return self.index.reconstruct(idx)

        def search(self, feature:Feature, k:int=3) -> Tuple[PairIDs, Scores]:
            """get feature(s) and IDs by proximity"""
            feature = feature[None].astype(np.float32) 
            scores, idxs = self.index.search(feature, k)
            # remove batch dim
            scores, idxs = scores[0], idxs[0]
            # remove -1 ids
            b = [i>=0 for i in idxs] 
            scores, idxs = scores[b], idxs[b]
            # map back to ids
            ids = [self.idx_to_id[i] for i in idxs]
            return ids, scores  
        
        def reset(self):
            self.index.reset()
            self.idx_to_id:Dict[int, PairID] = {}
            self.id_to_idx:Dict[PairID, int] = {}

        @property
        def ids(self):
            return self.id_to_idx.keys()

except ImportError:
    class IndexFastL2(Index):
        def __init__(self, *a, **kw):
            raise NotImplementedError("""install faiss for IndexFastL2""")

class NNSearch(JSONSerializable):
    """
    This class is the mid-level interface for neighbor search,
    providing some common utilities over the Index subclasses.
    Users will generally use `IML.search` instead of calling NNSearch directly.

    TODO: possibly get rid of this class and fold it into IML?
        * currently adds only complexity to the IML implementation
        * but could be useful if needing NNSearch without Feature/Interpolate?
    """
    def __init__(self, index:Index, k=10):
        """
        Args:
            index: instance of Index
            k: default k-nearest neighbors (but can be overridden later)
        """
        super().__init__(index=index, k=k)
        self.index = index
        self.default_k = k

    def __call__(self, feature:Feature, k:int=None) -> Tuple[PairIDs, Scores]:
        """
        find the k-nearest neighbors of `feature`
        Args:
            feature: query feature vector
            k: maximum number of neighbors to return
        Returns:
            ids: ids of neighbors
            scores: similarity scores of neighbors (higher is more similar)
        """
        k = k or self.default_k
        return self.index.search(feature, k)
    
    def distance(self, a:Feature, b:Feature):
        """compute distance between two features"""
        return self.index.metric(a, b)

    def add(self, feature: Feature, id:Optional[PairID]=None) -> PairID:
        """add a feature vector to the index and return its ID"""
        return self.index.add(feature, id)
    
    def get(self, id:PairID) -> Feature:
        """look up a feature by ID"""
        try:
            return self.index.get(id)
        except Exception:
            print(f"NNSearch: WARNING: can't `get` ID {id} which doesn't exist or has been removed")
    
    def remove(self, ids: Union[PairID, PairIDs]):
        """
        Remove point(s) from the index by ID
        """
        # iterable of ids case:
        if hasattr(ids, '__len__'):
            for id in ids:
                self.remove(id)
        # single id case:
        else:
            try:
                self.index.remove(ids)
            except Exception:
                print(f"NNSearch: WARNING: can't `remove` ID {ids} which doesn't exist or has already been removed")

    def remove_near(self, feature:Feature, k:int=None) -> PairIDs:
        """
        Remove point(s) from the index by proximity.
        Use k=1 to remove a single point.
        """
        k = k or self.default_k
        ids, _ = self(feature, k=k)
        self.remove(ids)
        return ids
    
    def reset(self):
        """clear all data from the index"""
        self.index.reset()

    def __iter__(self):
        """iterate over IDs in the index"""
        return iter(self.index.ids)
    
    def items(self) -> Generator[IDFeaturePair, None, None]:
        """iterate over ID, Feature pairs"""
        def iterator():
            for id in self.index.ids:
                yield IDFeaturePair(id, self.index.get(id))
        return iterator()

