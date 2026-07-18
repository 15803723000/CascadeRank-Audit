"""Tests for dataset loading and canonical graph preprocessing."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from cascaderank.data import LoadedGraph, load_dataset


class CustomCsvTests(unittest.TestCase):
    def _write_csv(self, directory: Path, text: str) -> Path:
        path = directory / "edges.csv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_loads_named_columns_and_preserves_isolated_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            path = self._write_csv(
                directory,
                "target,weight,source\n"
                "B,1,A\n"
                "A,2,B\n"
                "B,3,B\n"
                "C,4,C\n"
                "D,5,B\n",
            )

            loaded = load_dataset("custom", directory, path)

        self.assertIsInstance(loaded, LoadedGraph)
        self.assertEqual(loaded.name, "edges")
        self.assertEqual(loaded.node_ids, ["A", "B", "C", "D"])
        self.assertEqual(loaded.graph.number_of_nodes(), 4)
        self.assertEqual(
            {frozenset(edge) for edge in loaded.graph.edges()},
            {frozenset((0, 1)), frozenset((1, 3))},
        )
        self.assertEqual(loaded.graph.degree[2], 0)
        self.assertEqual(tuple(loaded.features.shape), (4, 6))
        self.assertTrue(torch.isfinite(loaded.features).all().item())
        self.assertEqual(loaded.edge_index.dtype, torch.long)
        self.assertEqual(
            set(map(tuple, loaded.edge_index.t().tolist())),
            {(0, 1), (1, 0), (1, 3), (3, 1)},
        )

    def test_uses_first_two_columns_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            path = self._write_csv(
                directory,
                "from,to,ignored\n10,20,x\n20,30,y\n",
            )

            first = load_dataset("my-network", directory, path)
            second = load_dataset("my-network", directory, path)

        self.assertEqual(first.name, "my-network")
        self.assertEqual(first.node_ids, ["10", "20", "30"])
        self.assertTrue(torch.equal(first.features, second.features))
        self.assertTrue(torch.equal(first.edge_index, second.edge_index))

    def test_rejects_malformed_or_unusable_csv_files(self) -> None:
        cases = {
            "empty": "",
            "one-column": "source\nA\n",
            "missing-cell": "source,target\nA,\n",
            "short-row": "source,target\nA\n",
            "only-self-loops": "source,target\nA,A\nB,B\n",
            "header-only": "source,target\n",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            for label, contents in cases.items():
                with self.subTest(label=label):
                    path = self._write_csv(directory, contents)
                    with self.assertRaisesRegex(ValueError, r".+"):
                        load_dataset("custom", directory, path)

    def test_rejects_unknown_dataset_without_csv(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported dataset"):
            load_dataset("unknown", Path("data"))

    def test_reports_missing_csv(self) -> None:
        missing = Path("definitely-not-present.csv")
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            load_dataset("custom", Path("data"), missing)


@unittest.skipUnless(
    importlib.util.find_spec("torch_geometric") is not None,
    "torch_geometric is not installed",
)
class PlanetoidTests(unittest.TestCase):
    @patch("torch_geometric.datasets.Planetoid")
    def test_canonicalizes_planetoid_graph(self, planetoid: MagicMock) -> None:
        data = SimpleNamespace(
            num_nodes=3,
            x=torch.tensor([[1, 0], [0, 1], [1, 1]]),
            edge_index=torch.tensor(
                [[0, 1, 1, 0, 2], [1, 0, 0, 0, 2]],
                dtype=torch.long,
            ),
        )
        pyg_dataset = MagicMock()
        pyg_dataset.__getitem__.return_value = data
        planetoid.return_value = pyg_dataset

        root = Path("cached-data")
        loaded = load_dataset("cOrA", root)

        planetoid.assert_called_once_with(root=str(root), name="Cora")
        self.assertEqual(loaded.name, "Cora")
        self.assertEqual(loaded.node_ids, [0, 1, 2])
        self.assertEqual(loaded.graph.number_of_nodes(), 3)
        self.assertEqual(loaded.graph.number_of_edges(), 1)
        self.assertEqual(tuple(loaded.features.shape), (3, 2))
        self.assertEqual(
            set(map(tuple, loaded.edge_index.t().tolist())),
            {(0, 1), (1, 0)},
        )


if __name__ == "__main__":
    unittest.main()
