import csv
from pathlib import Path

from cascaderank.pipeline import PipelineConfig, run_pipeline


def _write_bridge_graph(path: Path) -> None:
    edges = [
        ("a", "b"),
        ("b", "c"),
        ("c", "a"),
        ("d", "e"),
        ("e", "f"),
        ("f", "d"),
        ("c", "bridge"),
        ("bridge", "d"),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("source", "target"))
        writer.writerows(edges)


def test_custom_csv_pipeline_writes_all_artifacts(tmp_path: Path) -> None:
    edge_csv = tmp_path / "edges.csv"
    output_dir = tmp_path / "output"
    _write_bridge_graph(edge_csv)
    result = run_pipeline(
        PipelineConfig(
            dataset="CSV",
            edge_csv=edge_csv,
            data_root=tmp_path / "data",
            output_dir=output_dir,
            epochs=2,
            seed=7,
            report_mode="offline",
        )
    )

    assert result.rankings_path.is_file()
    assert result.attack_curves_path.read_bytes().startswith(b"\x89PNG")
    assert "结构洞" in result.explanation_path.read_text(encoding="utf-8")
    with result.rankings_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7
    for algorithm in (
        "gnn",
        "degree",
        "betweenness",
        "closeness",
        "eigenvector",
        "pagerank",
        "random",
    ):
        assert f"{algorithm}_score" in rows[0]
        assert f"{algorithm}_rank" in rows[0]
