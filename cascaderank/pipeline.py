"""End-to-end CascadeRank orchestration and command-line interface."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
import torch

from cascaderank.attack import AttackCurve
from cascaderank.attack import early_attack_diagnostics
from cascaderank.attack import plot_attack_curves
from cascaderank.attack import random_ranking
from cascaderank.attack import simulate_attack
from cascaderank.centrality import compute_centralities
from cascaderank.centrality import stable_rank_nodes
from cascaderank.data import LoadedGraph, load_dataset
from cascaderank.gnn import train_gat
from cascaderank.report import DEFAULT_MODEL, ReportResult, generate_explanation


ALGORITHMS = (
    "gnn",
    "degree",
    "betweenness",
    "closeness",
    "eigenvector",
    "pagerank",
    "random",
)


@dataclass(frozen=True)
class PipelineConfig:
    """Validated configuration for one CascadeRank run."""

    dataset: str = "Cora"
    edge_csv: Path | None = None
    data_root: Path = Path("data")
    output_dir: Path = Path(".")
    epochs: int = 50
    seed: int = 42
    hidden_channels: int = 8
    device: str = "auto"
    report_mode: str = "auto"
    openai_model: str = DEFAULT_MODEL
    approximate_betweenness_k: int | None = None


@dataclass(frozen=True)
class PipelineResult:
    """Artifacts and diagnostics produced by a completed pipeline."""

    rankings_path: Path
    attack_curves_path: Path
    explanation_path: Path
    early_attack: Mapping[str, float | bool]
    training_metadata: Mapping[str, Any]
    report: ReportResult
    timings: Mapping[str, float]


def _validate_config(config: PipelineConfig) -> None:
    if not config.dataset.strip():
        raise ValueError("dataset must not be empty")
    if not 1 <= config.epochs <= 50:
        raise ValueError("epochs must be between 1 and 50")
    if config.hidden_channels < 1:
        raise ValueError("hidden_channels must be positive")
    if config.report_mode not in {"auto", "openai", "offline"}:
        raise ValueError("invalid report mode")
    if not config.openai_model.startswith("gpt-5.6"):
        raise ValueError("openai_model must be in the GPT-5.6 family")


def _percentile_scores(scores: Mapping[int, float], node_count: int) -> np.ndarray:
    values = np.asarray([scores[node] for node in range(node_count)], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("centrality scores must be finite")
    if node_count == 1:
        return np.full(1, 0.5, dtype=float)

    order = np.argsort(values, kind="mergesort")
    percentiles = np.zeros(node_count, dtype=float)
    start = 0
    while start < node_count:
        end = start + 1
        while end < node_count and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        percentiles[order[start:end]] = average_rank / (node_count - 1)
        start = end
    return percentiles


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    minimum = float(values.min())
    maximum = float(values.max())
    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1.0e-12):
        return np.full_like(values, 0.5)
    return (values - minimum) / (maximum - minimum)


def _build_proxy_training_data(
    loaded: LoadedGraph,
    centralities: Mapping[str, Mapping[int, float]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Create transparent topology-proxy features and regression labels."""

    graph = loaded.graph
    node_count = graph.number_of_nodes()
    percentiles = {
        name: _percentile_scores(scores, node_count)
        for name, scores in centralities.items()
    }
    clustering_map = nx.clustering(graph)
    clustering = np.asarray(
        [clustering_map[node] for node in range(node_count)],
        dtype=float,
    )
    core_map = nx.core_number(graph)
    core = _minmax(
        np.asarray([core_map[node] for node in range(node_count)], dtype=float)
    )
    neighbor_degree_map = nx.average_neighbor_degree(graph)
    neighbor_degree = _minmax(
        np.asarray(
            [neighbor_degree_map[node] for node in range(node_count)],
            dtype=float,
        )
    )
    articulation_points = set(nx.articulation_points(graph))
    articulation = np.asarray(
        [1.0 if node in articulation_points else 0.0 for node in range(node_count)]
    )
    brokerage = _minmax(percentiles["betweenness"] * (1.0 - clustering))

    target = (
        0.30 * percentiles["betweenness"]
        + 0.18 * percentiles["degree"]
        + 0.12 * percentiles["pagerank"]
        + 0.08 * percentiles["closeness"]
        + 0.07 * percentiles["eigenvector"]
        + 0.15 * brokerage
        + 0.10 * articulation
    )
    target = _minmax(target)

    structural = np.column_stack(
        [
            percentiles["degree"],
            percentiles["betweenness"],
            percentiles["closeness"],
            percentiles["eigenvector"],
            percentiles["pagerank"],
            clustering,
            core,
            neighbor_degree,
            articulation,
            brokerage,
        ]
    ).astype(np.float32)
    base_features = loaded.features.detach().cpu().to(dtype=torch.float32)
    structural_tensor = torch.from_numpy(structural)
    features = torch.cat((base_features, structural_tensor), dim=1).contiguous()
    targets = torch.from_numpy(target.astype(np.float32))
    metadata = {
        "target_definition": {
            "betweenness_percentile": 0.30,
            "degree_percentile": 0.18,
            "pagerank_percentile": 0.12,
            "closeness_percentile": 0.08,
            "eigenvector_percentile": 0.07,
            "brokerage_proxy": 0.15,
            "articulation_indicator": 0.10,
        },
        "base_feature_count": int(base_features.shape[1]),
        "structural_feature_count": int(structural_tensor.shape[1]),
        "articulation_point_count": len(articulation_points),
    }
    return features, targets, metadata


def _node_score_mapping(scores: Sequence[float]) -> dict[int, float]:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("GNN scores must be a finite one-dimensional vector")
    return {node: float(value) for node, value in enumerate(values)}


def _rank_positions(ranking: Sequence[int]) -> dict[int, int]:
    return {node: position + 1 for position, node in enumerate(ranking)}


def _random_score_mapping(ranking: Sequence[int]) -> dict[int, float]:
    count = len(ranking)
    denominator = max(count - 1, 1)
    return {
        node: 1.0 - position / denominator
        for position, node in enumerate(ranking)
    }


def _safe_node_id(value: Any) -> str | int | float:
    if isinstance(value, (str, int, float)):
        return value
    return str(value)


def _write_rankings(
    path: Path,
    loaded: LoadedGraph,
    score_maps: Mapping[str, Mapping[int, float]],
    rankings: Mapping[str, Sequence[int]],
) -> None:
    rank_maps = {
        name: _rank_positions(ranking) for name, ranking in rankings.items()
    }
    headers = ["internal_node", "node_id"]
    for name in ALGORITHMS:
        headers.extend((f"{name}_score", f"{name}_rank"))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for node in range(loaded.graph.number_of_nodes()):
            row: dict[str, Any] = {
                "internal_node": node,
                "node_id": _safe_node_id(loaded.node_ids[node]),
            }
            for name in ALGORITHMS:
                row[f"{name}_score"] = f"{score_maps[name][node]:.12g}"
                row[f"{name}_rank"] = rank_maps[name][node]
            writer.writerow(row)


def _neighbor_group_count(graph: nx.Graph, node: int) -> int:
    neighbors = list(graph.neighbors(node))
    if not neighbors:
        return 0
    return nx.number_connected_components(graph.subgraph(neighbors))


def _topological_evidence(
    loaded: LoadedGraph,
    centralities: Mapping[str, Mapping[int, float]],
    gnn_scores: Mapping[int, float],
    gnn_ranking: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    graph = loaded.graph
    clustering = nx.clustering(graph)
    articulation_points = set(nx.articulation_points(graph))
    evidence: list[dict[str, Any]] = []

    candidate_pool = list(gnn_ranking[: min(100, len(gnn_ranking))])
    for rank, node in enumerate(candidate_pool, start=1):
        groups = _neighbor_group_count(graph, node)
        row = {
            "internal_node": node,
            "node_id": _safe_node_id(loaded.node_ids[node]),
            "gnn_rank": rank,
            "gnn_score": float(gnn_scores[node]),
            "degree": int(graph.degree(node)),
            "degree_centrality": float(centralities["degree"][node]),
            "betweenness": float(centralities["betweenness"][node]),
            "closeness": float(centralities["closeness"][node]),
            "eigenvector": float(centralities["eigenvector"][node]),
            "pagerank": float(centralities["pagerank"][node]),
            "clustering": float(clustering[node]),
            "neighbor_groups_without_node": groups,
            "is_articulation_point": node in articulation_points,
        }
        row["brokerage_proxy"] = row["betweenness"] * (
            1.0 - row["clustering"]
        )
        evidence.append(row)

    eligible = [
        row
        for row in evidence
        if row["degree"] >= 2 and row["neighbor_groups_without_node"] >= 2
    ]
    if not eligible:
        eligible = [row for row in evidence if row["degree"] >= 2]
    if not eligible:
        eligible = evidence
    if not eligible:
        raise RuntimeError("no node evidence was produced")

    candidate = max(
        eligible,
        key=lambda row: (
            bool(row["is_articulation_point"]),
            int(row["neighbor_groups_without_node"]),
            float(row["brokerage_proxy"]),
            -int(row["gnn_rank"]),
        ),
    )
    return evidence[:10], candidate


def _graph_summary(loaded: LoadedGraph) -> dict[str, Any]:
    graph = loaded.graph
    components = [len(component) for component in nx.connected_components(graph)]
    return {
        "dataset": loaded.name,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "connected_components": len(components),
        "largest_component_nodes": max(components),
        "is_connected": len(components) == 1,
        "input_feature_count": int(loaded.features.shape[1]),
        "graph_representation": "undirected_unweighted_simple",
        "self_loops_removed": True,
    }


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Execute the complete CascadeRank workflow and write three artifacts."""

    _validate_config(config)
    timings: dict[str, float] = {}
    total_start = time.perf_counter()
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_start = time.perf_counter()
    loaded = load_dataset(config.dataset, config.data_root, config.edge_csv)
    timings["load_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    centralities = compute_centralities(
        loaded.graph,
        seed=config.seed,
        approximate_betweenness_k=config.approximate_betweenness_k,
    )
    timings["centrality_seconds"] = time.perf_counter() - stage_start

    features, targets, proxy_metadata = _build_proxy_training_data(
        loaded,
        centralities,
    )
    stage_start = time.perf_counter()
    gnn_values, training_metadata = train_gat(
        features,
        loaded.edge_index,
        targets,
        epochs=config.epochs,
        seed=config.seed,
        hidden_channels=config.hidden_channels,
        device=config.device,
    )
    timings["training_seconds"] = time.perf_counter() - stage_start
    combined_training_metadata = dict(training_metadata)
    combined_training_metadata.update(proxy_metadata)

    gnn_scores = _node_score_mapping(gnn_values)
    score_maps: dict[str, Mapping[int, float]] = {
        "gnn": gnn_scores,
        **centralities,
    }
    rankings: dict[str, list[int]] = {
        name: stable_rank_nodes(scores) for name, scores in score_maps.items()
    }
    random_nodes = [int(node) for node in random_ranking(loaded.graph, config.seed)]
    rankings["random"] = random_nodes
    score_maps["random"] = _random_score_mapping(random_nodes)

    rankings_path = output_dir / "rankings.csv"
    _write_rankings(rankings_path, loaded, score_maps, rankings)

    stage_start = time.perf_counter()
    fractions = np.linspace(0.0, 1.0, 21)
    curves: dict[str, AttackCurve] = {
        "GNN": simulate_attack(loaded.graph, rankings["gnn"], fractions),
        "PageRank": simulate_attack(
            loaded.graph,
            rankings["pagerank"],
            fractions,
        ),
        "Random": simulate_attack(loaded.graph, rankings["random"], fractions),
    }
    attack_curves_path = output_dir / "attack_curves.png"
    plot_attack_curves(curves, attack_curves_path)
    diagnostics = early_attack_diagnostics(curves["GNN"], curves["Random"])
    timings["attack_and_plot_seconds"] = time.perf_counter() - stage_start

    top_nodes, structural_hole = _topological_evidence(
        loaded,
        centralities,
        gnn_scores,
        rankings["gnn"],
    )
    payload = {
        "top_nodes": top_nodes,
        "structural_hole_candidate": structural_hole,
        "early_attack_diagnostics": diagnostics,
        "training": combined_training_metadata,
        "centrality_approximation": {
            "betweenness": bool(
                config.approximate_betweenness_k is not None
                or loaded.graph.number_of_nodes() > 750
            ),
            "closeness": loaded.graph.number_of_nodes() > 5000,
        },
        "attack_protocol": {
            "fractions": curves["GNN"].fractions.tolist(),
            "gnn_lcc": curves["GNN"].largest_component_fractions.tolist(),
            "pagerank_lcc": (
                curves["PageRank"].largest_component_fractions.tolist()
            ),
            "random_lcc": curves["Random"].largest_component_fractions.tolist(),
            "random_seed": config.seed,
        },
    }
    explanation_path = output_dir / "explanation.txt"
    stage_start = time.perf_counter()
    report = generate_explanation(
        _graph_summary(loaded),
        payload,
        explanation_path,
        model=config.openai_model,
        mode=config.report_mode,
    )
    timings["report_seconds"] = time.perf_counter() - stage_start
    timings["total_seconds"] = time.perf_counter() - total_start

    return PipelineResult(
        rankings_path=rankings_path,
        attack_curves_path=attack_curves_path,
        explanation_path=explanation_path,
        early_attack=diagnostics,
        training_metadata=combined_training_metadata,
        report=report,
        timings=timings,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="CascadeRank Agent",
        description=(
            "Rank graph nodes with classical metrics and a two-layer GAT, "
            "then compare node-removal attacks."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="Cora",
        help="Cora, PubMed, CSV/custom, or a name paired with --edge-csv",
    )
    parser.add_argument(
        "--edge-csv",
        type=Path,
        help="custom edge-list CSV; source/target columns or the first two",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-channels", type=int, default=8)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
    )
    parser.add_argument(
        "--report-mode",
        default="auto",
        choices=("auto", "openai", "offline"),
        help=(
            "auto uses OpenAI when OPENAI_API_KEY exists; openai fails closed; "
            "offline writes a clearly labelled local report"
        ),
    )
    parser.add_argument("--openai-model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--betweenness-samples",
        type=int,
        help="override deterministic betweenness source-sample count",
    )
    return parser


def _print_result(result: PipelineResult) -> None:
    summary = {
        "rankings": str(result.rankings_path),
        "attack_curves": str(result.attack_curves_path),
        "explanation": str(result.explanation_path),
        "gnn_faster_than_random": result.early_attack[
            "gnn_faster_than_random"
        ],
        "report_provider": result.report.provider,
        "report_model": (
            result.report.model if result.report.provider == "openai" else None
        ),
        "response_id": result.report.response_id,
        "cached_tokens": result.report.cached_tokens,
        "cache_write_tokens": result.report.cache_write_tokens,
        "timings_seconds": {
            name: round(value, 3) for name, value in result.timings.items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = PipelineConfig(
        dataset=args.dataset,
        edge_csv=args.edge_csv,
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        seed=args.seed,
        hidden_channels=args.hidden_channels,
        device=args.device,
        report_mode=args.report_mode,
        openai_model=args.openai_model,
        approximate_betweenness_k=args.betweenness_samples,
    )
    try:
        result = run_pipeline(config)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"CascadeRank failed: {exc}", file=sys.stderr)
        return 1
    _print_result(result)
    return 0


__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "build_parser",
    "main",
    "run_pipeline",
]
