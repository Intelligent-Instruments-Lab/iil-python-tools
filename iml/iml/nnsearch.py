import numpy as np
from .types import *

try:
    import faiss
    from faiss import IndexFlatL2

except ImportError:

    class IndexFlatL2:
        def __init__(self, d):
            self.d = d
            
        def add(self, x):
            raise NotImplementedError("install faiss")

        def search(self, x, k):
            raise NotImplementedError("install faiss")

class NNSearch:
    def __init__(self, feature_size, index=IndexFlatL2):
        self.count = 0
        self.index = index(feature_size)

    def __call__(self, feature: Feature, k: int = 5) -> Tuple[TargetIDs, Scores]:
        """
        find the k-nearest neighbors of `feature`
        Args:
            feature: query feature vector
            k: maximum number of neighbors to return
        Returns:
            target_ids: ids of neighbors
            scores: similarity scores of neighbors (higher is more similar)
        """
        feature = feature[None].astype(np.float32) # add batch dim, enforce type
        scores, target_ids = self.index.search(feature, k)
        scores = scores[0] #** 0.5 # hardcoded for L2
        target_ids = target_ids[0] # remove batch dim
        b = [i>=0 for i in target_ids] # remove -1 ids
        target_ids = target_ids[b]
        scores = scores[b]
        return target_ids, scores
    
    def distance(self, a, b):
        # should match the index
        return sum((a-b) ** 2)

    # add a feature vector the the index and return its ID
    def add(self, feature: Feature) -> TargetID:
        feature = feature[None].astype(np.float32) 
        self.index.add(feature)
        target_id = self.count
        self.count = self.count+1
        return target_id

