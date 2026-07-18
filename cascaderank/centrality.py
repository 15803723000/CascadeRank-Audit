"""Classical graph-centrality calculations used by CascadeRank.

The public entry point returns the five centralities required by the CLI.  The
default policy is exact on small graphs and uses deterministic sampling where
an all-pairs traversal would make citation-sized graphs unnecessarily slow.
"""

from __future__ import annotations

import math
import random
import warnings
from collections.abc import Mapping
from numbers import Integral, Real

import networkx as nx


_CENTRALITY_NAMES = (
    "degree",
    "betweenness",
    "closeness",
    "eigenvector",
    "pagerank",
)
_EXACT_BETWEENNESS_MAX_NODES = 750
_MIN_BETWEENNESS_SAMPLES = 64
_MAX_BETWEENNESS_SAMPLES = 256
_EXACT_CLOSENESS_MAX_NODES = 5_000
_EXACT_CLOSENESS_COMPONENT_NODES = 512
_MIN_CLOSENESS_LANDMARKS = 64
_MAX_CLOSENESS_LANDMARKS = 192


def stable_rank_nodes(scores: Mapping[int, float]) -> list[int]:
    """Return node IDs ordered by descending score with stable integer ties.

    Scores must be finite.  Ties are broken by ascending node ID, so rankings
    do not depend on graph insertion order or Python dictionary ordering.
    """

    checked: list[tuple[int, float]] = []
    for node, score in scores.items():
        if isinstance(node, bool) or not isinstance(node, Integral):
            raise TypeError("centrality rankings require integer node IDs")
        if isinstance(score, bool) or not isinstance(score, Real):
            raise TypeError(f"score for node {node!r} is not numeric")
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError(f"score for node {node!r} is not finite")
        checked.append((int(node), numeric_score))

    if len({node for node, _ in checked}) != len(checked):
        raise ValueError("node IDs are not unique after integer conversion")

    checked.sort(key=lambda item: (-item[1], item[0]))
    return [node for node, _ in checked]


def compute_centralities(
    graph: nx.Graph,
    *,
    seed: int = 42,
    approximate_betweenness_k: int | None = None,
) -> dict[str, dict[int, float]]:
    """Compute five classical node-centrality score mappings.

    ``approximate_betweenness_k`` explicitly chooses the number of sampled
    sources.  When it is ``None``, graphs with at most 750 nodes are exact and
    larger graphs use a scale-aware, seeded sample.  Closeness follows a
    similar policy internally: it is exact through 5,000 nodes and otherwise
    estimates distance sums from seeded landmarks.  The latter keeps PubMed
    practical while retaining the disconnected-graph normalization used by
    :func:`networkx.closeness_centrality`.
    """

    _validate_inputs(graph, seed, approximate_betweenness_k)
    nodes = list(graph.nodes)
    if not nodes:
        return {name: {} for name in _CENTRALITY_NAMES}

    betweenness_k = _betweenness_sample_size(
        len(nodes), approximate_betweenness_k
    )
    raw_results: dict[str, Mapping[int, Real]] = {
        "degree": nx.degree_centrality(graph),
        "betweenness": nx.betweenness_centrality(
            graph,
            k=betweenness_k,
            normalized=True,
            weight=None,
            seed=int(seed),
        ),
        "closeness": _compute_closeness(graph, seed),
        "eigenvector": _compute_eigenvector(graph),
        "pagerank": _compute_pagerank(graph),
    }

    return {
        name: _complete_finite_scores(graph, name, raw_results[name])
        for name in _CENTRALITY_NAMES
    }


def _validate_inputs(
    graph: nx.Graph,
    seed: int,
    approximate_betweenness_k: int | None,
) -> None:
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a networkx Graph")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")

    normalized_nodes: list[int] = []
    for node in graph.nodes:
        if isinstance(node, bool) or not isinstance(node, Integral):
            raise TypeError("compute_centralities requires integer node IDs")
        normalized_nodes.append(int(node))
    if len(set(normalized_nodes)) != len(normalized_nodes):
        raise ValueError("node IDs are not unique after integer conversion")

    if approximate_betweenness_k is None:
        return
    if (
        isinstance(approximate_betweenness_k, bool)
        or not isinstance(approximate_betweenness_k, Integral)
    ):
        raise TypeError("approximate_betweenness_k must be an integer or None")
    if approximate_betweenness_k <= 0:
        raise ValueError("approximate_betweenness_k must be positive")
    if approximate_betweenness_k > graph.number_of_nodes():
        raise ValueError(
            "approximate_betweenness_k cannot exceed the node count"
        )


def _betweenness_sample_size(
    node_count: int,
    requested_k: int | None,
) -> int | None:
    if requested_k is not None:
        return int(requested_k)
    if node_count <= _EXACT_BETWEENNESS_MAX_NODES:
        return None
    scale_aware_k = math.ceil(math.sqrt(node_count))
    return min(
        node_count,
        _MAX_BETWEENNESS_SAMPLES,
        max(_MIN_BETWEENNESS_SAMPLES, scale_aware_k),
    )


def _compute_closeness(graph: nx.Graph, seed: int) -> dict[int, float]:
    node_count = graph.number_of_nodes()
    if node_count <= _EXACT_CLOSENESS_MAX_NODES:
        return dict(nx.closeness_centrality(graph, wf_improved=True))
    if graph.is_directed():
        return _approximate_directed_closeness(graph, seed)
    return _approximate_undirected_closeness(graph, seed)


def _approximate_undirected_closeness(
    graph: nx.Graph,
    seed: int,
) -> dict[int, float]:
    node_count = graph.number_of_nodes()
    results = {int(node): 0.0 for node in graph.nodes}
    generator = random.Random(int(seed))
    components = sorted(
        nx.connected_components(graph),
        key=lambda component: min(int(node) for node in component),
    )

    for component in components:
        ordered_nodes = sorted(component, key=int)
        component_size = len(ordered_nodes)
        if component_size <= 1:
            continue
        subgraph = graph.subgraph(ordered_nodes)
        global_scale = (component_size - 1) / (node_count - 1)

        if component_size <= _EXACT_CLOSENESS_COMPONENT_NODES:
            local_scores = nx.closeness_centrality(
                subgraph,
                wf_improved=True,
            )
            for node, score in local_scores.items():
                results[int(node)] = float(score) * global_scale
            continue

        landmark_count = _closeness_landmark_count(component_size)
        landmarks = generator.sample(ordered_nodes, landmark_count)
        sampled_distance_sums = {node: 0.0 for node in ordered_nodes}
        for landmark in landmarks:
            distances = nx.single_source_shortest_path_length(
                subgraph,
                landmark,
            )
            for node, distance in distances.items():
                sampled_distance_sums[node] += float(distance)

        expansion = component_size / landmark_count
        for node in ordered_nodes:
            estimated_distance_sum = (
                sampled_distance_sums[node] * expansion
            )
            if estimated_distance_sum <= 0.0:
                continue
            local_score = (component_size - 1) / estimated_distance_sum
            results[int(node)] = min(1.0, local_score * global_scale)

    return results


def _approximate_directed_closeness(
    graph: nx.Graph,
    seed: int,
) -> dict[int, float]:
    """Estimate NetworkX's default inward directed closeness."""

    ordered_nodes = sorted(graph.nodes, key=int)
    node_count = len(ordered_nodes)
    landmark_count = _closeness_landmark_count(node_count)
    landmarks = random.Random(int(seed)).sample(
        ordered_nodes,
        landmark_count,
    )
    reachable_samples = {node: 0 for node in ordered_nodes}
    sampled_distance_sums = {node: 0.0 for node in ordered_nodes}

    # NetworkX defines directed closeness with inward distances.  Running a
    # search from each sampled destination in the original graph obtains the
    # corresponding reverse-graph distance for every possible source node.
    for landmark in landmarks:
        distances = nx.single_source_shortest_path_length(graph, landmark)
        for node, distance in distances.items():
            reachable_samples[node] += 1
            sampled_distance_sums[node] += float(distance)

    expansion = node_count / landmark_count
    results: dict[int, float] = {}
    for node in ordered_nodes:
        estimated_reachable = min(
            float(node_count),
            reachable_samples[node] * expansion,
        )
        estimated_distance_sum = sampled_distance_sums[node] * expansion
        if estimated_reachable <= 1.0 or estimated_distance_sum <= 0.0:
            results[int(node)] = 0.0
            continue
        reachable_minus_one = estimated_reachable - 1.0
        score = reachable_minus_one / estimated_distance_sum
        score *= reachable_minus_one / (node_count - 1)
        results[int(node)] = min(1.0, score)
    return results


def _closeness_landmark_count(node_count: int) -> int:
    scale_aware_count = math.ceil(math.sqrt(node_count))
    return min(
        node_count,
        _MAX_CLOSENESS_LANDMARKS,
        max(_MIN_CLOSENESS_LANDMARKS, scale_aware_count),
    )


def _compute_eigenvector(graph: nx.Graph) -> dict[int, float]:
    nstart = {
        node: float(graph.degree(node)) + 1.0
        for node in graph.nodes
    }
    try:
        return dict(
            nx.eigenvector_centrality(
                graph,
                max_iter=500,
                tol=1.0e-8,
                nstart=nstart,
                weight=None,
            )
        )
    except nx.PowerIterationFailedConvergence:
        warnings.warn(
            "eigenvector centrality did not converge in 500 iterations; "
            "retrying with a relaxed tolerance and 5,000 iterations",
            RuntimeWarning,
            stacklevel=2,
        )

    try:
        return dict(
            nx.eigenvector_centrality(
                graph,
                max_iter=5_000,
                tol=1.0e-6,
                nstart=nstart,
                weight=None,
            )
        )
    except nx.PowerIterationFailedConvergence:
        warnings.warn(
            "eigenvector centrality failed its controlled retry; using a "
            "deterministic L2-normalized degree proxy",
            RuntimeWarning,
            stacklevel=2,
        )
        return _normalized_degree_fallback(graph)


def _normalized_degree_fallback(graph: nx.Graph) -> dict[int, float]:
    degrees = {node: float(graph.degree(node)) for node in graph.nodes}
    norm = math.sqrt(sum(value * value for value in degrees.values()))
    if norm > 0.0:
        return {int(node): value / norm for node, value in degrees.items()}
    uniform = 1.0 / math.sqrt(graph.number_of_nodes())
    return {int(node): uniform for node in graph.nodes}


def _compute_pagerank(graph: nx.Graph) -> dict[int, float]:
    try:
        return dict(
            nx.pagerank(
                graph,
                alpha=0.85,
                max_iter=200,
                tol=1.0e-10,
                weight=None,
            )
        )
    except nx.PowerIterationFailedConvergence:
        warnings.warn(
            "PageRank did not converge in 200 iterations; retrying with "
            "2,000 iterations and a relaxed tolerance",
            RuntimeWarning,
            stacklevel=2,
        )
        return dict(
            nx.pagerank(
                graph,
                alpha=0.85,
                max_iter=2_000,
                tol=1.0e-8,
                weight=None,
            )
        )


def _complete_finite_scores(
    graph: nx.Graph,
    name: str,
    scores: Mapping[int, Real],
) -> dict[int, float]:
    graph_nodes = list(graph.nodes)
    missing = [node for node in graph_nodes if node not in scores]
    extras = [node for node in scores if node not in graph]
    if missing or extras:
        raise RuntimeError(
            f"{name} returned an incomplete node mapping: "
            f"{len(missing)} missing, {len(extras)} unexpected"
        )

    checked: dict[int, float] = {}
    for node in graph_nodes:
        score = scores[node]
        if isinstance(score, bool) or not isinstance(score, Real):
            raise RuntimeError(f"{name} produced a non-numeric score")
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise RuntimeError(f"{name} produced a non-finite score")
        checked[int(node)] = numeric_score
    return checked
