"""Exact cosine retrieval without repeatedly normalizing or copying the index.

The encoder belongs to the caller so multiple memory systems can share one
SentenceTransformer.  Rebuilding changes the indexed text only when explicitly
requested; it does not refresh metadata after every memory evolution.
"""

from __future__ import annotations

from collections import OrderedDict
from operator import index
from typing import Iterable

import numpy as np
from sklearn.preprocessing import normalize


class FastEmbeddingRetriever:
    """An incrementally maintained, exact cosine index.

    ``document_ids`` preserves the upstream convention: duplicate texts remain
    separate corpus rows, and the mapping points to their last row.  Encodings
    are reused for identical text under an immutable encoder and configuration.
    The encoder must consistently return float32 or float64
    vectors, with one fixed dtype per index. Mixed precision normalization can
    change near-tie rankings and is rejected. ``embeddings`` contains normalized,
    active rows only.
    """

    def __init__(
        self,
        model,
        *,
        initial_capacity: int = 16,
        query_cache_size: int = 128,
        batch_size: int = 64,
    ):
        initial_capacity = index(initial_capacity)
        query_cache_size = index(query_cache_size)
        batch_size = index(batch_size)
        if initial_capacity < 0 or query_cache_size < 0 or batch_size <= 0:
            raise ValueError("capacity/cache size must be nonnegative; batch_size must be positive")
        self.model = model
        self.corpus: list[str] = []
        self.document_ids: dict[str, int] = {}
        self._buffer: np.ndarray | None = None
        self._size = 0
        self._dimension: int | None = None
        self._initial_capacity = initial_capacity
        self.query_cache_size = query_cache_size
        self.batch_size = batch_size
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.metrics = {
            "encode_calls": 0,
            "encode_texts": 0,
            "document_encode_calls": 0,
            "document_encode_texts": 0,
            "query_encode_calls": 0,
            "query_encode_texts": 0,
            "document_cache_hits": 0,
            "query_cache_hits": 0,
            "query_cache_misses": 0,
            "buffer_growths": 0,
            "buffer_reallocations": 0,
            "searches": 0,
            "partial_topk_calls": 0,
            "full_sort_fallbacks": 0,
        }

    @property
    def embeddings(self) -> np.ndarray | None:
        """A view of the normalized rows, excluding reserved buffer capacity."""
        if self._buffer is None:
            return None
        return self._buffer[: self._size]

    @property
    def capacity(self) -> int:
        return 0 if self._buffer is None else self._buffer.shape[0]

    @staticmethod
    def _documents(documents: Iterable[str]) -> list[str]:
        if isinstance(documents, str):
            raise TypeError("documents must be an iterable of strings, not one string")
        result = list(documents)
        if not all(isinstance(document, str) for document in result):
            raise TypeError("every document must be a string")
        return result

    def _encode(self, texts: list[str], kind: str) -> np.ndarray:
        self.metrics["encode_calls"] += 1
        self.metrics["encode_texts"] += len(texts)
        self.metrics[f"{kind}_encode_calls"] += 1
        self.metrics[f"{kind}_encode_texts"] += len(texts)
        # Explicitly chunk document batches in the caller.  Using encode(texts)
        # also permits minimal encoders in CPU-only experiments and unit tests.
        encoded = np.asarray(self.model.encode(texts))
        if encoded.ndim != 2 or encoded.shape[0] != len(texts) or encoded.shape[1] == 0:
            raise ValueError("encoder must return one nonempty vector per input text")
        if self._dimension is not None and encoded.shape[1] != self._dimension:
            raise ValueError("encoder embedding dimension changed")
        if encoded.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("encoder must return float32 or float64 vectors")
        if self._buffer is not None and encoded.dtype != self._buffer.dtype:
            raise ValueError("encoder embedding dtype changed; use one fixed dtype per index")
        # Match sklearn's cosine normalization, including zero and tiny vectors,
        # and preserve the encoder's fixed float32/float64 inference precision.
        return normalize(encoded, norm="l2", axis=1, copy=True)

    def _encode_missing(self, documents: list[str]) -> dict[str, np.ndarray]:
        missing = list(dict.fromkeys(doc for doc in documents if doc not in self.document_ids))
        result: dict[str, np.ndarray] = {}
        dimension = self._dimension
        dtype = self._buffer.dtype if self._buffer is not None else None
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors = self._encode(batch, "document")
            if dimension is None:
                dimension = vectors.shape[1]
            elif vectors.shape[1] != dimension:
                raise ValueError("encoder embedding dimension changed between batches")
            if dtype is None:
                dtype = vectors.dtype
            elif vectors.dtype != dtype:
                raise ValueError("encoder embedding dtype changed between batches")
            result.update(zip(batch, vectors))
        return result

    def _ensure_capacity(self, required: int, dimension: int, dtype) -> None:
        old_capacity = self.capacity
        dtype = np.dtype(dtype) if self._buffer is None else np.result_type(self._buffer.dtype, dtype)
        if self._buffer is not None and required <= old_capacity and dtype == self._buffer.dtype:
            return
        capacity = old_capacity or max(1, self._initial_capacity)
        while capacity < required:
            capacity *= 2
        replacement = np.empty((capacity, dimension), dtype=dtype)
        if self._buffer is not None:
            replacement[: self._size] = self._buffer[: self._size]
        self._buffer = replacement
        self._dimension = dimension
        self.metrics["buffer_reallocations"] += 1
        if capacity > old_capacity:
            self.metrics["buffer_growths"] += 1

    def add_documents(self, documents: Iterable[str]) -> None:
        documents = self._documents(documents)
        if not documents:
            return
        encoded = self._encode_missing(documents)
        example = next(iter(encoded.values()), None)
        if example is None:
            example = self._buffer[0]
        dtype = np.result_type(*(vector.dtype for vector in encoded.values())) if encoded else example.dtype
        self._ensure_capacity(self._size + len(documents), len(example), dtype)
        start = self._size
        for offset, document in enumerate(documents):
            vector = encoded.get(document)
            if vector is None:
                vector = self._buffer[self.document_ids[document]]
            self._buffer[start + offset] = vector
        self.corpus.extend(documents)
        self._size += len(documents)
        self.document_ids.update((document, start + offset) for offset, document in enumerate(documents))
        self.metrics["document_cache_hits"] += len(documents) - len(encoded)

    def replace_documents(self, documents: Iterable[str]) -> None:
        """Reconcile an ordered snapshot, encoding only previously absent text.

        Duplicate rows, removals, reordering, and unchanged rows retain the same
        meaning as a fresh upstream index.  The caller decides when metadata
        should become visible, e.g. at the original consolidation boundary.
        """
        documents = self._documents(documents)
        if not documents:
            self.corpus = []
            self.document_ids = {}
            self._size = 0
            return
        encoded = self._encode_missing(documents)
        changed = [
            position
            for position, document in enumerate(documents)
            if position >= self._size or self.corpus[position] != document
        ]
        # Save moved rows before writing, otherwise a permutation could overwrite
        # a source row needed by a later destination.
        moved = {
            documents[position]: self._buffer[self.document_ids[documents[position]]].copy()
            for position in changed
            if documents[position] not in encoded
        }
        example = next(iter(encoded.values()), None)
        if example is None:
            example = self._buffer[0]
        dtype = np.result_type(*(vector.dtype for vector in encoded.values())) if encoded else example.dtype
        self._ensure_capacity(len(documents), len(example), dtype)
        for position in changed:
            document = documents[position]
            self._buffer[position] = encoded[document] if document in encoded else moved[document]
        self.corpus = documents
        self._size = len(documents)
        self.document_ids = {document: position for position, document in enumerate(documents)}
        self.metrics["document_cache_hits"] += len(documents) - len(encoded)

    def rebuild(self, documents: Iterable[str]) -> None:
        self.replace_documents(documents)

    def consolidate(self, documents: Iterable[str]) -> None:
        self.replace_documents(documents)

    def reset(self) -> None:
        """Release document storage while retaining the model and query cache."""
        self.corpus = []
        self.document_ids = {}
        self._size = 0
        self._buffer = None
        self._dimension = None

    def clear_query_cache(self) -> None:
        self._query_cache.clear()

    def _query_embedding(self, query: str) -> np.ndarray:
        cached = self._query_cache.get(query)
        if cached is not None:
            self.metrics["query_cache_hits"] += 1
            self._query_cache.move_to_end(query)
            return cached
        self.metrics["query_cache_misses"] += 1
        vector = self._encode([query], "query")[0]
        if self.query_cache_size:
            self._query_cache[query] = vector
            if len(self._query_cache) > self.query_cache_size:
                self._query_cache.popitem(last=False)
        return vector

    def search(self, query: str, k: int = 5) -> np.ndarray:
        """Return row indices in exactly the upstream cosine ranking order.

        Partial selection handles distinct top scores in O(n + k log k).  NumPy
        quicksort has globally dependent ordering for ties, so ties within the
        result or at its boundary use the upstream full argsort expression.
        Nonpositive k returns no results (upstream accidentally returns all for
        k=0); values larger than the corpus return every row.
        """
        k = index(k)
        self.metrics["searches"] += 1
        if self._size == 0 or k <= 0:
            return np.empty(0, dtype=np.intp)
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        vector = self._query_embedding(query)
        if len(vector) != self._dimension:
            raise ValueError("cached query dimension differs from document embeddings")
        if vector.dtype != self._buffer.dtype:
            raise ValueError("cached query dtype differs from document embeddings; clear query cache")
        scores = np.dot(vector[np.newaxis, :], self.embeddings.T)[0]
        k = min(k, self._size)
        if k == self._size:
            self.metrics["full_sort_fallbacks"] += 1
            return np.argsort(scores)[-k:][::-1]
        self.metrics["partial_topk_calls"] += 1
        selected = np.argpartition(scores, self._size - k)[-k:]
        selected = selected[np.argsort(scores[selected])[::-1]]
        selected_scores = scores[selected]
        ties_inside = np.any(selected_scores[:-1] == selected_scores[1:])
        boundary_tied = np.count_nonzero(scores == selected_scores[-1]) > 1
        if ties_inside or boundary_tied:
            self.metrics["full_sort_fallbacks"] += 1
            return np.argsort(scores)[-k:][::-1]
        return selected


# Allow the optimized implementation to be substituted at the upstream name.
SimpleEmbeddingRetriever = FastEmbeddingRetriever
