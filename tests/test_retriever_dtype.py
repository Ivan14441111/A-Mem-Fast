"""Regression checks for the fixed-dtype exact-cosine contract."""


import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

import fast_amem.retriever as module


class Encoder:
    def __init__(self):
        self.dtype = np.float32

    def encode(self, texts):
        return np.asarray([[1, 2] if text != "q" else [2, 1] for text in texts], dtype=self.dtype)


def test_mixed_query_dtype_rejected_without_populating_query_cache():
    encoder = Encoder()
    retriever = module.FastEmbeddingRetriever(encoder)
    assert retriever.search("q").size == 0  # Empty/query-first operation stays lazy.
    retriever.add_documents(["document"])
    before = retriever.embeddings.copy()
    encoder.dtype = np.float64
    with pytest.raises(ValueError, match="dtype changed"):
        retriever.search("q")
    assert not retriever._query_cache
    assert retriever.corpus == ["document"]
    np.testing.assert_array_equal(retriever.embeddings, before)


@pytest.mark.parametrize("operation", ["add_documents", "rebuild"])
def test_document_or_rebuild_dtype_change_is_atomic(operation):
    encoder = Encoder()
    retriever = module.FastEmbeddingRetriever(encoder)
    retriever.add_documents(["old"])
    before = retriever.embeddings.copy()
    encoder.dtype = np.float64
    with pytest.raises(ValueError, match="dtype changed"):
        getattr(retriever, operation)(["new"])
    assert retriever.corpus == ["old"]
    np.testing.assert_array_equal(retriever.embeddings, before)


def test_initial_multibatch_dtype_change_does_not_create_partial_index():
    class AlternatingEncoder(Encoder):
        def encode(self, texts):
            result = super().encode(texts)
            self.dtype = np.float64
            return result

    retriever = module.FastEmbeddingRetriever(AlternatingEncoder(), batch_size=1)
    with pytest.raises(ValueError, match="dtype changed between batches"):
        retriever.add_documents(["first", "second"])
    assert retriever.corpus == []
    assert retriever.embeddings is None


def test_reset_with_changed_model_precision_requires_query_cache_clear():
    encoder = Encoder()
    retriever = module.FastEmbeddingRetriever(encoder)
    retriever.add_documents(["document"])
    retriever.search("q")
    retriever.reset()
    encoder.dtype = np.float64
    retriever.add_documents(["new document"])
    with pytest.raises(ValueError, match="cached query dtype"):
        retriever.search("q")
    retriever.clear_query_cache()
    np.testing.assert_array_equal(retriever.search("q"), [0])


@pytest.mark.parametrize("dtype", [np.float16, np.int64, np.complex64])
def test_unsupported_encoder_precision_is_explicitly_rejected(dtype):
    encoder = Encoder()
    encoder.dtype = dtype
    retriever = module.FastEmbeddingRetriever(encoder)
    with pytest.raises(ValueError, match="float32 or float64"):
        retriever.add_documents(["document"])
    assert retriever.embeddings is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_supported_fixed_dtype_near_ties_match_sklearn(dtype):
    rng = np.random.default_rng(7)
    documents = rng.normal(size=(12, 5)).astype(dtype)
    documents[1:6] = documents[0] + rng.normal(scale=1e-7, size=(5, 5)).astype(dtype)
    query = (documents[0] + rng.normal(scale=1e-6, size=5)).astype(dtype)

    class Model:
        def encode(self, texts):
            return query[None, :] if texts == ["q"] else documents[[int(text) for text in texts]]

    retriever = module.FastEmbeddingRetriever(Model())
    retriever.add_documents([str(i) for i in range(len(documents))])
    expected = np.argsort(cosine_similarity(query[None, :], documents)[0])[-5:][::-1]
    np.testing.assert_array_equal(retriever.search("q", 5), expected)
