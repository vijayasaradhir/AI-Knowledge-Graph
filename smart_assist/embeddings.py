from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(slots=True)
class SimilarityMatch:
    left_index: int
    right_index: int
    score: float


class TfidfEmbeddingEngine:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = None
        self._texts: List[str] = []

    def fit(self, texts: Sequence[str]) -> np.ndarray:
        self._texts = list(texts)
        self._matrix = self.vectorizer.fit_transform(self._texts)
        return self._matrix.toarray()

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self._matrix is None:
            raise RuntimeError("Embedding engine must be fit before transform.")
        matrix = self.vectorizer.transform(list(texts))
        return matrix.toarray()

    def similarity(self, left: str, right: str) -> float:
        vectors = self.vectorizer.transform([left, right])
        return float(cosine_similarity(vectors[0], vectors[1])[0][0])

    def nearest_neighbors(self, query: str, texts: Sequence[str], top_k: int = 3) -> List[SimilarityMatch]:
        matrix = self.vectorizer.transform(list(texts) + [query])
        sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
        ranked = sorted(
            (SimilarityMatch(left_index=i, right_index=len(texts), score=float(score)) for i, score in enumerate(sims)),
            key=lambda item: item.score,
            reverse=True,
        )
        return ranked[:top_k]

