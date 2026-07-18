"""Tests for classical centrality calculations."""

import math

import networkx as nx
import pytest

from cascaderank.centrality import compute_centralities
from cascaderank.centrality import stable_rank_nodes


EXPECTED_NAMES = {
    "degree",
    "betweenness",
    "closeness",
    "eigenvector",
    "pagerank",
}


def test_compute_centralities_is_complete_and_finite() -> None:
    graph = nx.path_graph(5)

    results = compute_centralities(graph)

    assert set(results) == EXPECTED_NAMES
    for scores in results.values():
        assert set(scores) == set(graph.nodes)
        assert all(math.isfinite(score) for score in scores.values())
    assert results["degree"][1] == pytest.approx(0.5)
    assert results["betweenness"][2] == pytest.approx(2.0 / 3.0)
    assert results["closeness"][2] == pytest.approx(2.0 / 3.0)


def test_empty_and_disconnected_graphs_are_supported() -> None:
    empty_results = compute_centralities(nx.Graph())
    assert empty_results == {name: {} for name in EXPECTED_NAMES}

    graph = nx.Graph([(0, 1), (2, 3)])
    graph.add_node(4)
    results = compute_centralities(graph)

    for scores in results.values():
        assert set(scores) == set(graph.nodes)
        assert all(math.isfinite(score) for score in scores.values())
    assert results["closeness"][4] == 0.0


def test_sampled_betweenness_is_reproducible() -> None:
    graph = nx.barabasi_albert_graph(40, 2, seed=9)

    first = compute_centralities(
        graph,
        seed=17,
        approximate_betweenness_k=10,
    )
    second = compute_centralities(
        graph,
        seed=17,
        approximate_betweenness_k=10,
    )

    assert first["betweenness"] == second["betweenness"]


def test_stable_rank_nodes_breaks_ties_by_integer_id() -> None:
    scores = {8: 0.25, 3: 0.9, 5: 0.25, 1: 0.9}

    assert stable_rank_nodes(scores) == [1, 3, 5, 8]


def test_invalid_sampling_count_is_rejected() -> None:
    graph = nx.path_graph(4)

    with pytest.raises(ValueError):
        compute_centralities(graph, approximate_betweenness_k=5)


def test_eigenvector_failure_uses_explicit_fallback(monkeypatch) -> None:
    graph = nx.path_graph(4)

    def fail_to_converge(*args, **kwargs):
        raise nx.PowerIterationFailedConvergence(1)

    monkeypatch.setattr(nx, "eigenvector_centrality", fail_to_converge)
    with pytest.warns(RuntimeWarning) as caught:
        results = compute_centralities(graph)

    assert any("degree proxy" in str(item.message) for item in caught)
    assert set(results["eigenvector"]) == set(graph.nodes)
    assert all(
        math.isfinite(score)
        for score in results["eigenvector"].values()
    )
