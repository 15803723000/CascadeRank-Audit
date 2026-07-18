"""Focused tests for the CascadeRank GAT proxy regressor."""

import numpy as np
import pytest
import torch

from cascaderank.gnn import GATRanker, train_gat


def _small_graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        [
            [1.0, 0.0, 0.2],
            [0.8, 0.2, 0.1],
            [0.0, 1.0, 0.4],
            [0.1, 0.9, 0.3],
        ]
    )
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3],
        ],
        dtype=torch.long,
    )
    targets = torch.tensor([0.9, 0.6, 0.3, 0.1])
    return features, edge_index, targets


def test_ranker_outputs_one_bounded_score_per_node() -> None:
    features, edge_index, _ = _small_graph()
    model = GATRanker(in_channels=3, hidden_channels=4, heads=2, dropout=0.0)
    model.eval()
    assert model.gat1.res is not None
    assert model.gat2.res is not None

    with torch.no_grad():
        scores = model(features, edge_index)

    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()
    assert torch.all((scores >= 0.0) & (scores <= 1.0))


def test_training_is_deterministic_and_returns_metadata() -> None:
    features, edge_index, targets = _small_graph()
    first_scores, first_metadata = train_gat(
        features,
        edge_index,
        targets,
        epochs=3,
        seed=7,
        hidden_channels=4,
        device="cpu",
    )
    second_scores, second_metadata = train_gat(
        features,
        edge_index,
        targets,
        epochs=3,
        seed=7,
        hidden_channels=4,
        device="cpu",
    )

    np.testing.assert_allclose(first_scores, second_scores, atol=0.0, rtol=0.0)
    assert first_scores.shape == (4,)
    assert np.isfinite(first_scores).all()
    assert np.all((first_scores >= 0.0) & (first_scores <= 1.0))
    assert first_metadata == second_metadata
    assert len(first_metadata["loss_history"]) == 3
    assert first_metadata["epochs"] == 3
    assert first_metadata["device"] == "cpu"
    assert first_metadata["objective"] == "topology_proxy_regression"
    assert first_metadata["architecture"] == "two_layer_residual_gat"


def test_single_node_with_no_edges_is_supported() -> None:
    scores, metadata = train_gat(
        torch.tensor([[1.0, 0.0]]),
        torch.empty((2, 0), dtype=torch.long),
        torch.tensor([1.0]),
        epochs=1,
        device="cpu",
    )

    np.testing.assert_array_equal(scores, np.array([0.5]))
    assert metadata["num_nodes"] == 1
    assert metadata["num_edges"] == 0


@pytest.mark.parametrize("epochs", [0, 51])
def test_epoch_limit_is_enforced(epochs: int) -> None:
    features, edge_index, targets = _small_graph()

    with pytest.raises(ValueError, match="between 1 and 50"):
        train_gat(features, edge_index, targets, epochs=epochs)


def test_invalid_edge_shape_and_index_are_rejected() -> None:
    features, _, targets = _small_graph()

    with pytest.raises(ValueError, match="shape"):
        train_gat(features, torch.tensor([0, 1]), targets, epochs=1)
    with pytest.raises(ValueError, match="outside"):
        train_gat(
            features,
            torch.tensor([[0, 4], [1, 2]]),
            targets,
            epochs=1,
        )
