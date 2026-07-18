from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from cascaderank.attack import early_attack_diagnostics
from cascaderank.attack import plot_attack_curves
from cascaderank.attack import simulate_attack


def test_simulate_attack_uses_original_node_denominator() -> None:
    graph = nx.path_graph(5)
    fractions = [0.0, 0.2, 0.4, 1.0]
    curve = simulate_attack(graph, [2, 1, 3, 0, 4], fractions)

    assert curve.removed_counts.tolist() == [0, 1, 2, 5]
    assert curve.largest_component_fractions.tolist() == pytest.approx(
        [1.0, 0.4, 0.4, 0.0]
    )


def test_attack_rejects_incomplete_ranking() -> None:
    with pytest.raises(ValueError, match="every graph node"):
        simulate_attack(nx.path_graph(3), [0, 1])


def test_plot_and_early_diagnostic(tmp_path: Path) -> None:
    graph = nx.path_graph(20)
    fractions = np.linspace(0.0, 1.0, 21)
    targeted = simulate_attack(graph, list(range(1, 20, 2)) + [0] + list(
        range(2, 20, 2)
    ), fractions)
    random_like = simulate_attack(graph, list(range(20)), fractions)
    diagnostics = early_attack_diagnostics(targeted, random_like)
    assert "gnn_faster_than_random" in diagnostics

    output = tmp_path / "curves.png"
    plot_attack_curves(
        {"GNN": targeted, "PageRank": random_like, "Random": random_like},
        output,
    )
    assert output.read_bytes().startswith(b"\x89PNG")
