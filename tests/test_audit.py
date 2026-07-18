import networkx as nx

from cascaderank.audit import AuditConfig, collective_influence_scores
from cascaderank.audit import detect_signal_overlap
from cascaderank.audit import evaluate_ranking_audit, run_audit


def test_detect_signal_overlap_identifies_direct_leakage() -> None:
    findings = detect_signal_overlap(
        ["pagerank_percentile", "degree_percentile"],
        ["degree", "pagerank", "clustering"],
        ["pagerank", "betweenness"],
    )
    assert {item["signal"] for item in findings} == {"degree", "pagerank"}
    assert any(item["kind"] == "label_feature_overlap" for item in findings)
    assert any(item["kind"] == "target_baseline_overlap" for item in findings)


def test_evaluate_ranking_audit_reports_unsupported_proxy_claim() -> None:
    graph = nx.path_graph(6)
    rankings = {
        "proxy_gat": [0, 1, 2, 3, 4, 5],
        "teacher": [0, 1, 2, 3, 4, 5],
        "degree": [2, 3, 1, 4, 0, 5],
        "betweenness": [2, 3, 1, 4, 0, 5],
        "closeness": [2, 3, 1, 4, 0, 5],
        "eigenvector": [2, 3, 1, 4, 0, 5],
        "pagerank": [2, 3, 1, 4, 0, 5],
    }
    result = evaluate_ranking_audit(
        graph,
        rankings,
        target_signals=["degree", "pagerank"],
        feature_signals=["degree", "pagerank"],
        random_trials=5,
        seed=3,
    )

    assert result["best_traditional_baseline"] in {
        "degree",
        "betweenness",
        "closeness",
        "eigenvector",
        "pagerank",
    }
    assert result["claims"][0]["verdict"] == "NOT_SUPPORTED"
    assert result["claims"][1]["verdict"] == "INCONCLUSIVE"
    assert result["claims"][2]["verdict"] == "SUPPORTED"
    assert len(result["random"]["mean_curve"]) == 21


def test_topology_only_target_has_no_declared_baseline_overlap() -> None:
    findings = detect_signal_overlap(
        ["single_node_lcc_loss"],
        [],
        [
            "degree",
            "betweenness",
            "closeness",
            "eigenvector",
            "pagerank",
            "collective_influence",
        ],
    )
    assert findings == []


def test_collective_influence_prioritizes_a_bridge() -> None:
    graph = nx.barbell_graph(4, 2)
    scores = collective_influence_scores(graph)
    assert max(scores, key=scores.get) in {3, 4, 5, 6}


def test_audit_run_writes_hashed_artifacts(tmp_path) -> None:
    edge_csv = tmp_path / "graph.csv"
    edge_csv.write_text(
        "source,target\na,b\nb,c\nc,d\nd,a\nb,e\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "audit"
    result = run_audit(
        AuditConfig(
            dataset="CSV",
            edge_csv=edge_csv,
            output_dir=output_dir,
            mode="topology-only",
            epochs=2,
            random_trials=3,
        )
    )
    assert result.manifest_path.is_file()
    assert result.report_path.is_file()
    assert result.html_path.is_file()
    assert result.chart_path.is_file()
    assert result.evidence["leakage_findings"] == []
    assert "collective_influence" in result.evidence["metrics"]
