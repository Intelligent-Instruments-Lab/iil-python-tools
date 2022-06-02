import numpy as np
import faiss

from .types import *

class NNSearch:
    def __init__(self, feature_size, index=faiss.IndexFlatL2):
        self.count = 0
        self.index = index(feature_size)

    def __call__(self, feature: Feature, k: int = 5) -> Tuple[TargetIDs, Scores]:
        feature = feature[None].astype(np.float32) # add batch dim, enforce type
        scores, target_ids = self.index.search(feature, k)
        scores = scores ** 0.5 # hardcoded for L2
        return target_ids[0], scores[0] # remove batch dim

    # add a feature vector the the index and return its ID
    def add(self, feature: Feature) -> TargetID:
        feature = feature[None].astype(np.float32) 
        self.index.add(feature)
        target_id = self.count
        self.count = self.count+1
        return target_id