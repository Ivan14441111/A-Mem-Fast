"""CPU-only retrieval equivalence and incremental-maintenance checks."""

import math

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from fast_amem.retriever import FastEmbeddingRetriever, SimpleEmbeddingRetriever


class FakeEncoder:
    def __init__(self, vectors, dtype=np.float64):
        self.vectors = vectors
        self.dtype = dtype
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.asarray([self.vectors[text] for text in texts], dtype=self.dtype)


def expected(model, documents, query, k):
    raw_documents = np.asarray([model.vectors[doc] for doc in documents], dtype=model.dtype)
    raw_query = np.asarray([model.vectors[query]], dtype=model.dtype)
    scores = cosine_similarity(raw_query, raw_documents)[0]
    return np.argsort(scores)[-k:][::-1]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("count,k", [(1, 1), (2, 1), (31, 5), (101, 13), (50, 50), (13, 100)])
def test_exact_ranking_matches_upstream_cosine(dtype, count, k):
    rng = np.random.default_rng(7053)
    vectors = {f"d{i}": rng.normal(size=23) for i in range(count)}
    vectors["query"] = rng.normal(size=23)
    model = FakeEncoder(vectors, dtype)
    retriever = FastEmbeddingRetriever(model, batch_size=7)
    documents = list(vectors)[:-1]
    retriever.add_documents(documents)
    np.testing.assert_array_equal(retriever.search("query", k), expected(model, documents, "query", k))
    assert retriever.model is model
    np.testing.assert_allclose(np.linalg.norm(retriever.embeddings, axis=1), 1, atol=2e-7)


@pytest.mark.parametrize("count", [5, 17, 65, 129])
@pytest.mark.parametrize("k", [1, 3, 5])
def test_all_ties_keep_numpy_quicksort_order(count, k):
    # Tie order can vary with corpus length and cannot be reproduced by merely
    # sorting the rows that argpartition happens to select.
    vectors = {f"d{i}": [0, 1, 0] for i in range(count)}
    vectors["query"] = [1, 0, 0]
    model = FakeEncoder(vectors)
    retriever = FastEmbeddingRetriever(model)
    documents = list(vectors)[:-1]
    retriever.add_documents(documents)
    np.testing.assert_array_equal(retriever.search("query", k), expected(model, documents, "query", k))
    assert retriever.metrics["full_sort_fallbacks"] == 1


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 7])
def test_ties_inside_topk_and_at_boundary(k):
    values = [0, 1, 0.5, 1, -0.8, 0.5, -1, 0.3, 0.3, 0.2]
    vectors = {f"d{i}": [value, np.sqrt(1 - value**2)] for i, value in enumerate(values)}
    vectors["query"] = [1, 0]
    model = FakeEncoder(vectors)
    documents = list(vectors)[:-1]
    retriever = FastEmbeddingRetriever(model)
    retriever.add_documents(documents)
    np.testing.assert_array_equal(retriever.search("query", k), expected(model, documents, "query", k))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_tiny_and_negative_vectors_match_sklearn(dtype):
    tiny = np.finfo(dtype).eps / 100
    vectors = {"zero": [0, 0], "tiny": [tiny, 0], "right": [2, 0], "left": [-3, 0], "q": [1, 0]}
    model = FakeEncoder(vectors, dtype)
    retriever = FastEmbeddingRetriever(model)
    documents = ["zero", "tiny", "right", "left"]
    retriever.add_documents(documents)
    np.testing.assert_array_equal(retriever.search("q", 3), expected(model, documents, "q", 3))
    np.testing.assert_array_equal(retriever.embeddings[0], [0, 0])
    assert retriever.embeddings[1, 0] == tiny


def test_geometric_growth_and_active_embedding_view():
    vectors = {str(i): [i + 1, 1] for i in range(129)}
    model = FakeEncoder(vectors)
    retriever = FastEmbeddingRetriever(model, initial_capacity=1)
    previous_buffer = None
    previous_capacity = 0
    for i in range(129):
        retriever.add_documents([str(i)])
        assert retriever.embeddings.shape == (i + 1, 2)
        assert retriever.capacity >= i + 1
        if i + 1 <= previous_capacity:
            assert retriever._buffer is previous_buffer
        previous_buffer = retriever._buffer
        previous_capacity = retriever.capacity
    assert retriever.capacity == 256
    assert retriever.metrics["buffer_growths"] == math.ceil(math.log2(129)) + 1
    assert retriever.metrics["document_encode_texts"] == 129


def test_batching_duplicates_and_last_document_id():
    model = FakeEncoder({str(i): [i + 1, 1] for i in range(9)})
    retriever = FastEmbeddingRetriever(model, batch_size=3)
    documents = ["0", "1", "0", "2", "3", "4", "5", "6", "7", "8", "8"]
    retriever.add_documents(documents)
    assert model.calls == [["0", "1", "2"], ["3", "4", "5"], ["6", "7", "8"]]
    assert retriever.corpus == documents
    assert retriever.document_ids["0"] == 2
    assert retriever.document_ids["8"] == 10
    assert retriever.metrics["document_encode_texts"] == 9
    assert retriever.metrics["document_cache_hits"] == 2
    retriever.add_documents(["1", "1"])
    assert len(model.calls) == 3
    assert retriever.document_ids["1"] == 12
    np.testing.assert_array_equal(retriever.embeddings[1], retriever.embeddings[-1])


def test_corpus_does_not_alias_input_list():
    model = FakeEncoder({"a": [1, 0], "b": [0, 1]})
    retriever = FastEmbeddingRetriever(model)
    documents = ["a"]
    retriever.add_documents(documents)
    documents.append("b")
    assert retriever.corpus == ["a"]
    retriever.rebuild(documents)
    documents.clear()
    assert retriever.corpus == ["a", "b"]


def test_query_lru_cache_has_bounded_size_and_true_recency():
    vectors = {"doc": [1, 0], "q1": [1, 1], "q2": [0, 1], "q3": [-1, 1]}
    model = FakeEncoder(vectors)
    retriever = FastEmbeddingRetriever(model, query_cache_size=2)
    retriever.add_documents(["doc"])
    for query in ["q1", "q2", "q1", "q3", "q1", "q2"]:
        retriever.search(query)
    assert model.calls == [["doc"], ["q1"], ["q2"], ["q3"], ["q2"]]
    assert len(retriever._query_cache) == 2
    assert retriever.metrics["query_cache_hits"] == 2
    assert retriever.metrics["query_cache_misses"] == 4
    retriever.clear_query_cache()
    retriever.search("q2")
    assert retriever.metrics["query_encode_calls"] == 5


def test_query_cache_can_be_disabled():
    model = FakeEncoder({"doc": [1, 0], "q": [1, 1]})
    retriever = FastEmbeddingRetriever(model, query_cache_size=0)
    retriever.add_documents(["doc"])
    retriever.search("q")
    retriever.search("q")
    assert retriever.metrics["query_encode_calls"] == 2
    assert retriever.metrics["query_cache_hits"] == 0
    assert not retriever._query_cache


def test_rebuild_handles_permutations_duplicates_additions_and_removal():
    vectors = {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1], "new": [1, 1, 1], "q": [1, 2, 3]}
    model = FakeEncoder(vectors)
    retriever = FastEmbeddingRetriever(model)
    retriever.add_documents(["a", "b", "c"])
    original_buffer = retriever._buffer
    retriever.search("q")
    replacement = ["c", "a", "b", "new", "a"]
    retriever.rebuild(replacement)
    assert model.calls == [["a", "b", "c"], ["q"], ["new"]]
    assert retriever.model is model
    assert retriever._buffer is original_buffer
    assert retriever.corpus == replacement
    assert retriever.document_ids == {"c": 0, "a": 4, "b": 2, "new": 3}
    np.testing.assert_array_equal(retriever.search("q", 4), expected(model, replacement, "q", 4))
    retriever.replace_documents(["b", "c"])
    np.testing.assert_array_equal(retriever.search("q", 2), expected(model, ["b", "c"], "q", 2))
    assert retriever.embeddings.shape == (2, 3)
    assert retriever.metrics["document_encode_texts"] == 4
    assert retriever.metrics["query_encode_texts"] == 1


def test_repeated_consolidation_encodes_only_changed_metadata_texts():
    vectors = {
        "content:a context:x": [1, 0],
        "a , x keyword tag": [1, 1],
        "b , x keyword tag": [1, -1],
        "b , changed keyword tag": [0, 1],
    }
    model = FakeEncoder(vectors)
    retriever = FastEmbeddingRetriever(model)
    retriever.add_documents(["content:a context:x"])
    retriever.consolidate(["a , x keyword tag", "b , x keyword tag"])
    calls_after_first = len(model.calls)
    before = retriever.embeddings.copy()
    retriever.consolidate(["a , x keyword tag", "b , x keyword tag"])
    assert len(model.calls) == calls_after_first
    np.testing.assert_array_equal(retriever.embeddings, before)
    retriever.consolidate(["a , x keyword tag", "b , changed keyword tag"])
    assert model.calls[-1] == ["b , changed keyword tag"]
    np.testing.assert_array_equal(retriever.embeddings[0], before[0])


def test_empty_rebuild_reset_and_reuse_keep_model():
    model = FakeEncoder({"a": [1, 0], "b": [0, 1], "q": [1, 1]})
    retriever = FastEmbeddingRetriever(model)
    retriever.add_documents(["a"])
    retriever.search("q")
    retriever.rebuild([])
    assert retriever.corpus == []
    assert retriever.document_ids == {}
    assert retriever.embeddings.shape == (0, 2)
    assert retriever.search("q").size == 0
    retriever.add_documents(["b"])
    retriever.search("q")
    assert retriever.metrics["query_encode_calls"] == 1
    retriever.reset()
    assert retriever.embeddings is None
    assert retriever.capacity == 0
    assert retriever.model is model
    retriever.add_documents(["a"])
    retriever.search("q")
    assert retriever.metrics["query_encode_calls"] == 1


def test_empty_and_nonpositive_k_never_encode_query():
    model = FakeEncoder({"a": [1, 0]})
    retriever = FastEmbeddingRetriever(model)
    assert retriever.search("not in encoder").size == 0
    retriever.add_documents([])
    assert model.calls == []
    assert retriever.embeddings is None
    retriever.add_documents(["a"])
    assert retriever.search("not in encoder", 0).size == 0
    assert retriever.search("not in encoder", -1).size == 0
    assert retriever.metrics["query_encode_calls"] == 0
    with pytest.raises(TypeError):
        retriever.search("a", 1.5)


@pytest.mark.parametrize("kwargs", [{"initial_capacity": -1}, {"query_cache_size": -1}, {"batch_size": 0}])
def test_invalid_constructor_configuration(kwargs):
    with pytest.raises(ValueError):
        FastEmbeddingRetriever(FakeEncoder({}), **kwargs)


@pytest.mark.parametrize("documents", ["one document", [1], ["a", None]])
def test_invalid_document_input_is_rejected(documents):
    retriever = FastEmbeddingRetriever(FakeEncoder({}))
    with pytest.raises(TypeError):
        retriever.add_documents(documents)
    assert retriever.corpus == []


def test_failed_rebuild_leaves_existing_index_valid():
    model = FakeEncoder({"a": [1, 0], "bad": [np.nan, 0], "q": [1, 1]})
    retriever = FastEmbeddingRetriever(model)
    retriever.add_documents(["a"])
    before = retriever.embeddings.copy()
    with pytest.raises(ValueError):
        retriever.rebuild(["bad"])
    assert retriever.corpus == ["a"]
    np.testing.assert_array_equal(retriever.embeddings, before)
    np.testing.assert_array_equal(retriever.search("q"), [0])


def test_dimension_change_is_rejected_without_mutating_corpus():
    model = FakeEncoder({"a": [1, 0], "other": [1, 0, 0]})
    retriever = FastEmbeddingRetriever(model)
    retriever.add_documents(["a"])
    with pytest.raises(ValueError, match="dimension"):
        retriever.add_documents(["other"])
    assert retriever.corpus == ["a"]
    assert SimpleEmbeddingRetriever is FastEmbeddingRetriever
