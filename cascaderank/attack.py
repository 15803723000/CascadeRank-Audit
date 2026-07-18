"""Node-removal attack simulation and plotting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Mapping, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

plt.switch_backend("Agg")


@dataclass(frozen=True)
class AttackCurve:
    """Largest-connected-component measurements for one node ranking."""

    fractions: np.ndarray
    largest_component_fractions: np.ndarray
    removed_counts: np.ndarray


def _validate_fractions(fractions: Sequence[float]) -> np.ndarray:
    values = np.asarray(fractions, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("fractions must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("fractions must contain only finite values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("fractions must lie in the closed interval [0, 1]")
    if np.any(np.diff(values) < 0.0):
        raise ValueError("fractions must be sorted in non-decreasing order")
    return values


def _validate_ranking(
    graph: nx.Graph,
    ranking: Sequence[Hashable],
) -> list[Hashable]:
    nodes = set(graph.nodes)
    ordered = list(ranking)
    if len(ordered) != len(nodes):
        raise ValueError("ranking must contain every graph node exactly once")
    try:
        ordered_nodes = set(ordered)
    except TypeError as exc:
        raise ValueError("ranking contains an unhashable node") from exc
    if len(ordered_nodes) != len(ordered) or ordered_nodes != nodes:
        raise ValueError("ranking must contain every graph node exactly once")
    return ordered


def _largest_component_size(graph: nx.Graph) -> int:
    if graph.number_of_nodes() == 0:
        return 0
    return max(len(component) for component in nx.connected_components(graph))


def simulate_attack(
    graph: nx.Graph,
    ranking: Sequence[Hashable],
    fractions: Sequence[float] | None = None,
) -> AttackCurve:
    """Remove ranked nodes cumulatively and measure the largest component.

    The denominator is the original total node count, so all strategies share a
    common, interpretable scale even when the input graph is disconnected.
    """

    if graph.is_directed():
        raise ValueError("attack simulation requires an undirected graph")
    node_count = graph.number_of_nodes()
    if node_count == 0:
        raise ValueError("attack simulation requires at least one node")

    values = _validate_fractions(
        np.linspace(0.0, 1.0, 21) if fractions is None else fractions
    )
    ordered = _validate_ranking(graph, ranking)
    removed_counts = np.floor(values * node_count + 1e-12).astype(int)
    removed_counts = np.clip(removed_counts, 0, node_count)

    working = graph.copy()
    component_fractions: list[float] = []
    previous_count = 0
    for removal_count in removed_counts:
        if removal_count > previous_count:
            working.remove_nodes_from(ordered[previous_count:removal_count])
            previous_count = int(removal_count)
        component_fractions.append(_largest_component_size(working) / node_count)

    return AttackCurve(
        fractions=values,
        largest_component_fractions=np.asarray(component_fractions),
        removed_counts=removed_counts,
    )


def random_ranking(graph: nx.Graph, seed: int = 42) -> list[Hashable]:
    """Return a deterministic random permutation of the graph nodes."""

    nodes = list(graph.nodes)
    generator = np.random.default_rng(seed)
    generator.shuffle(nodes)
    return nodes


def early_attack_diagnostics(
    gnn_curve: AttackCurve,
    random_curve: AttackCurve,
) -> dict[str, float | bool]:
    """Quantify the pre-specified 5% and 10% dismantling comparison."""

    checkpoints = (0.05, 0.10)
    clear_gap_threshold = 0.05
    result: dict[str, float | bool] = {}
    gaps: list[float] = []
    for checkpoint in checkpoints:
        gnn_value = float(
            np.interp(
                checkpoint,
                gnn_curve.fractions,
                gnn_curve.largest_component_fractions,
            )
        )
        random_value = float(
            np.interp(
                checkpoint,
                random_curve.fractions,
                random_curve.largest_component_fractions,
            )
        )
        label = str(int(checkpoint * 100))
        result[f"gnn_lcc_at_{label}pct"] = gnn_value
        result[f"random_lcc_at_{label}pct"] = random_value
        result[f"gap_at_{label}pct"] = random_value - gnn_value
        gaps.append(random_value - gnn_value)

    result["mean_gap_5_to_10pct"] = float(np.mean(gaps))
    result["clear_gap_threshold"] = clear_gap_threshold
    result["gnn_faster_than_random"] = bool(
        all(gap >= clear_gap_threshold for gap in gaps)
    )
    return result


def plot_attack_curves(
    curves: Mapping[str, AttackCurve],
    output_path: Path,
) -> None:
    """Render exactly the three required strategy curves to a PNG file."""

    expected = ("GNN", "PageRank", "Random")
    if tuple(curves.keys()) != expected:
        raise ValueError(f"curves must be ordered and named exactly {expected}")

    styles = {
        "GNN": {"color": "#c0392b", "marker": "o"},
        "PageRank": {"color": "#2471a3", "marker": "s"},
        "Random": {"color": "#626567", "marker": "^"},
    }
    figure, axis = plt.subplots(figsize=(9.0, 5.8), constrained_layout=True)
    for label, curve in curves.items():
        axis.plot(
            curve.fractions,
            curve.largest_component_fractions,
            label=label,
            linewidth=2.2,
            markersize=4.0,
            markevery=2,
            **styles[label],
        )

    axis.axvspan(0.05, 0.10, color="#f5b7b1", alpha=0.22)
    axis.set(
        xlabel="Fraction of nodes removed",
        ylabel="Largest connected component / original nodes",
        title="CascadeRank targeted node-removal attack",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
    )
    axis.grid(alpha=0.22, linewidth=0.8)
    axis.legend(frameon=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, metadata={"Software": "CascadeRank"})
    plt.close(figure)
