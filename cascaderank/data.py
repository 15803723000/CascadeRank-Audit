"""Dataset loading and graph preprocessing for CascadeRank."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import torch


@dataclass(slots=True)
class LoadedGraph:
    """Canonical in-memory representation used by the ranking pipeline."""

    name: str
    graph: nx.Graph
    features: torch.Tensor
    edge_index: torch.Tensor
    node_ids: list[Any]


def load_dataset(
    dataset: str,
    data_root: Path,
    edge_csv: Path | None = None,
) -> LoadedGraph:
    """Load a Planetoid graph or a custom edge-list CSV.

    Built-in names are case-insensitive.  A custom graph can be requested with
    ``dataset="custom"`` (or ``"csv"``) and ``edge_csv``.  Supplying
    ``edge_csv`` with any other non-built-in name uses that name for the
    returned dataset, which is useful for named custom datasets.
    """

    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be a non-empty string")

    requested_name = dataset.strip()
    normalized_name = requested_name.casefold()
    planetoid_names = {"cora": "Cora", "pubmed": "PubMed"}

    if normalized_name in planetoid_names:
        if edge_csv is not None:
            raise ValueError(
                "edge_csv cannot be combined with a built-in dataset"
            )
        return _load_planetoid(
            planetoid_names[normalized_name], Path(data_root)
        )

    if normalized_name in {"custom", "csv", "edge_csv"}:
        if edge_csv is None:
            raise ValueError(
                "a custom dataset requires --edge-csv (edge_csv argument)"
            )
        csv_path = Path(edge_csv)
        return _load_edge_csv(csv_path, csv_path.stem)

    if edge_csv is not None:
        return _load_edge_csv(Path(edge_csv), requested_name)

    raise ValueError(
        f"unsupported dataset {requested_name!r}; choose Cora, PubMed, "
        "or provide edge_csv for a custom graph"
    )


def _load_planetoid(name: str, data_root: Path) -> LoadedGraph:
    try:
        from torch_geometric.datasets import Planetoid
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError(
            "loading Cora or PubMed requires the torch_geometric package"
        ) from exc

    root = data_root.expanduser()
    if root.exists() and not root.is_dir():
        raise ValueError(f"Planetoid data root is not a directory: {root}")

    try:
        pyg_dataset = Planetoid(root=str(root), name=name)
        data = pyg_dataset[0]
    except Exception as exc:
        raise RuntimeError(
            f"failed to load Planetoid dataset {name!r} under "
            f"{root / name}: {exc}"
        ) from exc

    num_nodes = int(data.num_nodes or 0)
    if num_nodes == 0:
        raise ValueError(f"Planetoid dataset {name!r} contains no nodes")

    raw_edge_index = data.edge_index
    if raw_edge_index is None or raw_edge_index.ndim != 2:
        raise ValueError(f"Planetoid dataset {name!r} has invalid edge_index")
    if raw_edge_index.shape[0] != 2:
        raise ValueError(f"Planetoid dataset {name!r} has invalid edge_index")

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    sources = raw_edge_index[0].detach().cpu().tolist()
    targets = raw_edge_index[1].detach().cpu().tolist()
    for source, target in zip(sources, targets):
        source = int(source)
        target = int(target)
        if not 0 <= source < num_nodes or not 0 <= target < num_nodes:
            raise ValueError(
                f"Planetoid dataset {name!r} contains an out-of-range edge"
            )
        if source != target:
            graph.add_edge(source, target)

    _require_usable_graph(graph, f"Planetoid dataset {name!r}")

    if data.x is None:
        features = _structural_features(graph)
    else:
        if data.x.ndim != 2 or data.x.shape[0] != num_nodes:
            raise ValueError(
                f"Planetoid dataset {name!r} has incompatible features"
            )
        features = data.x.detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()

    return LoadedGraph(
        name=name,
        graph=graph,
        features=features,
        edge_index=_edge_index_from_graph(graph),
        node_ids=list(range(num_nodes)),
    )


def _load_edge_csv(path: Path, name: str) -> LoadedGraph:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"edge-list CSV does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"edge-list CSV is not a file: {path}")

    node_to_index: dict[str, int] = {}
    node_ids: list[str] = []
    graph = nx.Graph()

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header, header_line = _next_non_empty_row(reader)
            if header is None:
                raise ValueError(f"edge-list CSV is empty: {path}")
            if len(header) < 2:
                raise ValueError(
                    f"edge-list CSV must have at least two columns: {path}"
                )

            source_index, target_index = _edge_column_indices(header)
            last_index = max(source_index, target_index)
            for line_number, row in enumerate(
                reader, start=header_line + 1
            ):
                if not row or not any(cell.strip() for cell in row):
                    continue
                if len(row) <= last_index:
                    raise ValueError(
                        f"row {line_number} in {path} has fewer columns "
                        "than its header"
                    )
                source_id = row[source_index].strip()
                target_id = row[target_index].strip()
                if not source_id or not target_id:
                    raise ValueError(
                        f"row {line_number} in {path} has an empty node ID"
                    )

                source = _intern_node(
                    source_id, node_to_index, node_ids, graph
                )
                target = _intern_node(
                    target_id, node_to_index, node_ids, graph
                )
                if source != target:
                    graph.add_edge(source, target)
    except (OSError, UnicodeError, csv.Error) as exc:
        message = f"could not read edge-list CSV {path}: {exc}"
        raise ValueError(message) from exc

    if not node_ids:
        raise ValueError(f"edge-list CSV contains no data rows: {path}")
    _require_usable_graph(graph, f"edge-list CSV {path}")

    return LoadedGraph(
        name=name,
        graph=graph,
        features=_structural_features(graph),
        edge_index=_edge_index_from_graph(graph),
        node_ids=node_ids,
    )


def _next_non_empty_row(
    reader: csv.reader,
) -> tuple[list[str] | None, int]:
    for line_number, row in enumerate(reader, start=1):
        if row and any(cell.strip() for cell in row):
            return row, line_number
    return None, 0


def _edge_column_indices(header: list[str]) -> tuple[int, int]:
    normalized = [column.strip().casefold() for column in header]
    if "source" in normalized and "target" in normalized:
        return normalized.index("source"), normalized.index("target")
    return 0, 1


def _intern_node(
    node_id: str,
    node_to_index: dict[str, int],
    node_ids: list[str],
    graph: nx.Graph,
) -> int:
    index = node_to_index.get(node_id)
    if index is not None:
        return index
    index = len(node_ids)
    node_to_index[node_id] = index
    node_ids.append(node_id)
    graph.add_node(index)
    return index


def _require_usable_graph(graph: nx.Graph, description: str) -> None:
    if graph.number_of_nodes() == 0:
        raise ValueError(f"{description} contains no nodes")
    if graph.number_of_edges() == 0:
        raise ValueError(
            f"{description} contains no usable edges after removing self-loops"
        )


def _edge_index_from_graph(graph: nx.Graph) -> torch.Tensor:
    undirected_edges = sorted(
        (min(int(source), int(target)), max(int(source), int(target)))
        for source, target in graph.edges()
        if source != target
    )
    if not undirected_edges:
        return torch.empty((2, 0), dtype=torch.long)

    sources: list[int] = []
    targets: list[int] = []
    for source, target in undirected_edges:
        sources.extend((source, target))
        targets.extend((target, source))
    return torch.tensor((sources, targets), dtype=torch.long)


def _structural_features(graph: nx.Graph) -> torch.Tensor:
    """Create six bounded structural features without a dense adjacency."""

    num_nodes = graph.number_of_nodes()
    scale = max(num_nodes - 1, 1)
    log_scale = math.log1p(scale)
    degrees = dict(graph.degree())
    neighbor_degrees = nx.average_neighbor_degree(graph)
    clustering = nx.clustering(graph)
    core_numbers = nx.core_number(graph)

    component_fraction: dict[int, float] = {}
    for component in nx.connected_components(graph):
        fraction = len(component) / num_nodes
        for node in component:
            component_fraction[int(node)] = fraction

    rows: list[list[float]] = []
    for node in range(num_nodes):
        degree = float(degrees[node])
        rows.append(
            [
                degree / scale,
                math.log1p(degree) / log_scale,
                float(neighbor_degrees[node]) / scale,
                float(clustering[node]),
                float(core_numbers[node]) / scale,
                component_fraction[node],
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


__all__ = ["LoadedGraph", "load_dataset"]
