"""Graph-attention ranking model and deterministic training utilities.

The model in this module learns to regress supplied topological proxy scores.
It does not, by itself, identify causal influence or provide an unsupervised
definition of node criticality.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GATConv


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


class GATRanker(nn.Module):
    """A compact two-layer GAT that emits one score per node.

    The first layer uses configurable multi-head attention. The second layer
    has one non-concatenated head and one output channel. Both layers use the
    standard GAT residual path so node-local structural evidence is not erased
    by neighborhood aggregation. A sigmoid constrains the model output to the
    unit interval.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 8,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if isinstance(in_channels, bool) or not isinstance(in_channels, int):
            raise TypeError("in_channels must be an integer")
        if isinstance(hidden_channels, bool) or not isinstance(
            hidden_channels, int
        ):
            raise TypeError("hidden_channels must be an integer")
        if isinstance(heads, bool) or not isinstance(heads, int):
            raise TypeError("heads must be an integer")
        if in_channels < 1:
            raise ValueError("in_channels must be at least 1")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be at least 1")
        if heads < 1:
            raise ValueError("heads must be at least 1")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
            raise TypeError("dropout must be a real number")
        if not math.isfinite(float(dropout)) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")

        self.in_channels = in_channels
        self.dropout = float(dropout)
        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=self.dropout,
            add_self_loops=True,
            residual=True,
        )
        self.gat2 = GATConv(
            hidden_channels * heads,
            1,
            heads=1,
            concat=False,
            dropout=self.dropout,
            add_self_loops=True,
            residual=True,
        )

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return one finite-range score for each input node."""
        _validate_features(features, expected_channels=self.in_channels)
        _validate_edge_index(edge_index, features.size(0))
        if features.device != edge_index.device:
            raise ValueError("features and edge_index must be on one device")

        parameter_dtype = next(self.parameters()).dtype
        if features.dtype != parameter_dtype:
            features = features.to(dtype=parameter_dtype)
        if edge_index.dtype != torch.long:
            edge_index = edge_index.to(dtype=torch.long)

        hidden = self.gat1(features, edge_index)
        hidden = torch.nn.functional.elu(hidden)
        hidden = torch.nn.functional.dropout(
            hidden,
            p=self.dropout,
            training=self.training,
        )
        logits = self.gat2(hidden, edge_index).squeeze(-1)
        return torch.sigmoid(logits)


def train_gat(
    features: torch.Tensor,
    edge_index: torch.Tensor,
    target_scores: torch.Tensor,
    *,
    epochs: int = 50,
    seed: int = 42,
    hidden_channels: int = 8,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    device: str = "auto",
) -> tuple[
    np.ndarray,
    dict[str, list[float] | float | int | str],
]:
    """Fit a GAT to supplied topological proxy scores.

    All nodes are used for proxy-score regression; this is intentionally a
    ranking surrogate rather than a held-out predictive evaluation. Returned
    scores are monotonically min-max normalized to ``[0, 1]``. A constant
    prediction, including the single-node case, maps to ``0.5``.
    """
    _validate_training_options(
        epochs=epochs,
        seed=seed,
        hidden_channels=hidden_channels,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    _validate_features(features)
    _validate_edge_index(edge_index, features.size(0))
    targets = _prepare_targets(target_scores, features.size(0))
    selected_device = _resolve_device(device)
    _seed_everything(seed)

    model = GATRanker(
        in_channels=features.size(1),
        hidden_channels=hidden_channels,
    ).to(selected_device)
    model_features = features.detach().to(
        device=selected_device,
        dtype=torch.float32,
    )
    model_edges = edge_index.detach().to(
        device=selected_device,
        dtype=torch.long,
    )
    model_targets = targets.detach().to(
        device=selected_device,
        dtype=torch.float32,
    )
    normalized_targets = _minmax_normalize_tensor(model_targets)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    criterion = nn.MSELoss()
    loss_history: list[float] = []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predictions = model(model_features, model_edges)
        loss = criterion(predictions, normalized_targets)
        if not torch.isfinite(loss):
            raise RuntimeError("GAT training produced a non-finite loss")
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().cpu().item()))

    model.eval()
    with torch.no_grad():
        learned_scores = model(model_features, model_edges)
        learned_scores = _minmax_normalize_tensor(learned_scores)

    scores = learned_scores.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(scores).all():
        raise RuntimeError("GAT inference produced non-finite scores")
    scores = np.clip(scores, 0.0, 1.0)
    metadata: dict[str, list[float] | float | int | str] = {
        "loss_history": loss_history,
        "final_loss": loss_history[-1],
        "device": str(selected_device),
        "epochs": epochs,
        "seed": seed,
        "num_nodes": int(features.size(0)),
        "num_edges": int(edge_index.size(1)),
        "objective": "topology_proxy_regression",
        "architecture": "two_layer_residual_gat",
        "score_normalization": "monotonic_minmax",
    }
    return scores, metadata


def _validate_features(
    features: torch.Tensor,
    expected_channels: int | None = None,
) -> None:
    if not isinstance(features, torch.Tensor):
        raise TypeError("features must be a torch.Tensor")
    if features.ndim != 2:
        raise ValueError("features must have shape [num_nodes, num_features]")
    if features.size(0) < 1:
        raise ValueError("features must contain at least one node")
    if features.size(1) < 1:
        raise ValueError("features must contain at least one feature")
    if expected_channels is not None and features.size(1) != expected_channels:
        raise ValueError(
            f"expected {expected_channels} features per node, "
            f"received {features.size(1)}"
        )
    if features.dtype == torch.bool or not (
        features.is_floating_point() or features.dtype in _INTEGER_DTYPES
    ):
        raise TypeError("features must have a real numeric dtype")
    if not torch.isfinite(features).all().item():
        raise ValueError("features must contain only finite values")


def _validate_edge_index(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> None:
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError("edge_index must be a torch.Tensor")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_index.dtype not in _INTEGER_DTYPES:
        raise TypeError("edge_index must use an integer dtype")
    if edge_index.numel() == 0:
        return
    minimum = int(edge_index.min().item())
    maximum = int(edge_index.max().item())
    if minimum < 0 or maximum >= num_nodes:
        raise ValueError(
            "edge_index contains a node index outside the feature matrix"
        )


def _prepare_targets(
    target_scores: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    if not isinstance(target_scores, torch.Tensor):
        raise TypeError("target_scores must be a torch.Tensor")
    if target_scores.ndim == 2 and target_scores.shape == (num_nodes, 1):
        target_scores = target_scores.squeeze(1)
    if target_scores.ndim != 1 or target_scores.size(0) != num_nodes:
        raise ValueError("target_scores must have shape [num_nodes] or [N, 1]")
    if target_scores.dtype == torch.bool or not (
        target_scores.is_floating_point()
        or target_scores.dtype in _INTEGER_DTYPES
    ):
        raise TypeError("target_scores must have a real numeric dtype")
    if not torch.isfinite(target_scores).all().item():
        raise ValueError("target_scores must contain only finite values")
    return target_scores


def _validate_training_options(**options: Any) -> None:
    epochs = options["epochs"]
    seed = options["seed"]
    hidden_channels = options["hidden_channels"]
    learning_rate = options["learning_rate"]
    weight_decay = options["weight_decay"]

    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError("epochs must be an integer")
    if not 1 <= epochs <= 50:
        raise ValueError("epochs must be between 1 and 50")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be in [0, 2**32 - 1]")
    if isinstance(hidden_channels, bool) or not isinstance(
        hidden_channels, int
    ):
        raise TypeError("hidden_channels must be an integer")
    if hidden_channels < 1:
        raise ValueError("hidden_channels must be at least 1")
    _validate_nonnegative_float(
        learning_rate,
        name="learning_rate",
        allow_zero=False,
    )
    _validate_nonnegative_float(
        weight_decay,
        name="weight_decay",
        allow_zero=True,
    )


def _validate_nonnegative_float(
    value: Any,
    *,
    name: str,
    allow_zero: bool,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    numeric_value = float(value)
    lower_bound_satisfied = (
        numeric_value >= 0.0 if allow_zero else numeric_value > 0.0
    )
    if not math.isfinite(numeric_value) or not lower_bound_satisfied:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")


def _resolve_device(device: str) -> torch.device:
    if not isinstance(device, str):
        raise TypeError("device must be a string")
    requested = device.strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        selected = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid device: {device!r}") from exc
    if selected.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'mps'")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if selected.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return selected


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _minmax_normalize_tensor(values: torch.Tensor) -> torch.Tensor:
    minimum = values.min()
    maximum = values.max()
    value_range = maximum - minimum
    tolerance = torch.finfo(values.dtype).eps
    if float(value_range.detach().cpu().item()) <= tolerance:
        return torch.full_like(values, 0.5)
    return ((values - minimum) / value_range).clamp(0.0, 1.0)
