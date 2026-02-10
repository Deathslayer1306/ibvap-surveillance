import faiss
import numpy as np
import os
from typing import Tuple, cast

INDEX_PATH = "face_index.faiss"


class FaceVectorDB:
    def __init__(self, dim: int = 512):
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
        else:
            self.index = faiss.IndexFlatIP(dim)

    def search(self, embedding: np.ndarray, threshold: float = 0.6):
        if self.index.ntotal == 0:
            return None

        vec = np.asarray(embedding, dtype="float32")[None, :]

        sims, ids = cast(
            Tuple[np.ndarray, np.ndarray],
            self.index.search(vec, 1),
        )

        if sims[0][0] >= threshold:
            return int(ids[0][0])

        return None

    def add(self, embedding: np.ndarray) -> int:
        vec = np.asarray(embedding, dtype="float32")[None, :]
        cast(object, self.index).add(vec)
        faiss.write_index(self.index, INDEX_PATH)
        return self.index.ntotal - 1
