"""Evidence-first audit workflow for graph node-ranking claims.

This module deliberately separates deterministic measurements from the language
used to describe them.  It can therefore report that a learned ranker did *not*
add value instead of forcing a favorable GNN narrative.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from cascaderank.attack import AttackCurve, random_ranking, simulate_attack
from cascaderank.centrality import compute_centralities, stable_rank_nodes
from cascaderank.data import load_dataset
from cascaderank.gnn import train_gat
from cascaderank.pipeline import _minmax


DEFAULT_FRACTIONS = tuple(float(value) for value in np.linspace(0.0, 1.0, 21))
VERDICTS = {"SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE"}


@dataclass(frozen=True)
class AuditConfig:
    """Configuration for one reproducible CascadeRank Audit run."""

    dataset: str = "Cora"
    edge_csv: Path | None = None
    data_root: Path = Path("data")
    output_dir: Path = Path("audit_output")
    mode: str = "leaky"
    epochs: int = 30
    seed: int = 42
    random_trials: int = 100
    hidden_channels: int = 8
    device: str = "auto"


@dataclass(frozen=True)
class AuditRunResult:
    """Paths and machine-readable evidence produced by an audit run."""

    manifest_path: Path
    report_path: Path
    html_path: Path
    chart_path: Path
    evidence: Mapping[str, Any]


def _signal_name(value: str) -> str:
    """Normalize metric labels before comparing declared provenance."""

    normalized = value.strip().casefold().replace("_", "-")
    for suffix in ("-percentile", "-indicator", "-score"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized


def detect_signal_overlap(
    target_signals: Sequence[str],
    feature_signals: Sequence[str],
    baseline_signals: Sequence[str],
) -> list[dict[str, str]]:
    """Return auditable overlap findings between labels, inputs, and baselines."""

    target = {_signal_name(value) for value in target_signals}
    features = {_signal_name(value) for value in feature_signals}
    baselines = {_signal_name(value) for value in baseline_signals}
    signals = sorted(target | features | baselines)
    findings: list[dict[str, str]] = []
    for signal in signals:
        if signal in target and signal in features:
            findings.append(
                {
                    "signal": signal,
                    "kind": "label_feature_overlap",
                    "severity": "error",
                    "message": (
                        "The same signal is used to construct the training label "
                        "and supplied as a model feature."
                    ),
                }
            )
        if signal in target and signal in baselines:
            findings.append(
                {
                    "signal": signal,
                    "kind": "target_baseline_overlap",
                    "severity": "warning",
                    "message": (
                        "This baseline contributes to the teacher target and is "
                        "therefore not an independent comparison."
                    ),
                }
            )
    return findings


def _curve_metrics(curve: AttackCurve) -> dict[str, float]:
    """Summarize a curve with pre-registered checkpoints and area."""

    values = curve.largest_component_fractions
    fractions = curve.fractions
    return {
        "lcc_at_5pct": float(np.interp(0.05, fractions, values)),
        "lcc_at_10pct": float(np.interp(0.10, fractions, values)),
        # NumPy 1.x exposes ``trapz`` while NumPy 2.x also exposes
        # ``trapezoid``.  Keep the audit runnable on either supported line.
        "attack_auc": float(np.trapz(values, fractions)),
    }


def rank_spearman(
    first: Sequence[int], second: Sequence[int]
) -> float:
    """Compute Spearman correlation for two complete, untied node rankings."""

    if len(first) != len(second) or set(first) != set(second):
        raise ValueError("rankings must contain the same nodes exactly once")
    count = len(first)
    if count < 2:
        return 1.0
    second_positions = {node: index for index, node in enumerate(second)}
    first_positions = np.arange(count, dtype=float)
    aligned_second = np.asarray(
        [second_positions[node] for node in first], dtype=float
    )
    correlation = np.corrcoef(first_positions, aligned_second)[0, 1]
    return float(correlation)


def collective_influence_scores(
    graph: nx.Graph,
    radius: int = 2,
) -> dict[int, float]:
    """Compute the fixed-radius Collective Influence structural baseline.

    CI is deliberately a non-learned comparator. It scores a node by its excess
    degree times the excess-degree mass exactly ``radius`` hops away.
    """

    if radius < 1:
        raise ValueError("collective influence radius must be positive")
    degrees = dict(graph.degree())
    scores: dict[int, float] = {}
    for node in graph.nodes:
        distances = nx.single_source_shortest_path_length(
            graph, node, cutoff=radius
        )
        boundary_mass = sum(
            degrees[other] - 1
            for other, distance in distances.items()
            if distance == radius
        )
        scores[node] = float((degrees[node] - 1) * boundary_mass)
    return scores


def evaluate_ranking_audit(
    graph: nx.Graph,
    rankings: Mapping[str, Sequence[int]],
    *,
    proxy_name: str = "proxy_gat",
    teacher_name: str = "teacher",
    baseline_names: Sequence[str] = (
        "degree",
        "betweenness",
        "closeness",
        "eigenvector",
        "pagerank",
        "collective_influence",
    ),
    target_signals: Sequence[str] = (),
    feature_signals: Sequence[str] = (),
    random_trials: int = 100,
    seed: int = 42,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
) -> dict[str, Any]:
    """Measure rankings, random uncertainty, provenance conflicts, and claims."""

    if graph.number_of_nodes() < 2:
        raise ValueError("audit requires a graph with at least two nodes")
    if random_trials < 2:
        raise ValueError("random_trials must be at least two")
    if proxy_name not in rankings or teacher_name not in rankings:
        raise ValueError("rankings must include proxy and teacher rankings")

    fraction_values = np.asarray(fractions, dtype=float)
    deterministic_curves = {
        name: simulate_attack(graph, ranking, fraction_values)
        for name, ranking in rankings.items()
    }
    deterministic_metrics = {
        name: _curve_metrics(curve)
        for name, curve in deterministic_curves.items()
    }

    random_values = np.asarray(
        [
            simulate_attack(
                graph,
                random_ranking(graph, seed + trial),
                fraction_values,
            ).largest_component_fractions
            for trial in range(random_trials)
        ],
        dtype=float,
    )
    random_mean_curve = AttackCurve(
        fractions=fraction_values,
        largest_component_fractions=random_values.mean(axis=0),
        removed_counts=np.floor(
            fraction_values * graph.number_of_nodes() + 1.0e-12
        ).astype(int),
    )
    random_metrics = _curve_metrics(random_mean_curve)
    available_baselines = [
        name for name in baseline_names if name in deterministic_metrics
    ]
    if not available_baselines:
        raise ValueError("at least one traditional baseline must be present")
    best_baseline = min(
        available_baselines,
        key=lambda name: deterministic_metrics[name]["attack_auc"],
    )
    proxy_metrics = deterministic_metrics[proxy_name]
    best_metrics = deterministic_metrics[best_baseline]
    overlap = detect_signal_overlap(
        target_signals,
        feature_signals,
        available_baselines,
    )
    direct_leakage = any(
        item["kind"] == "label_feature_overlap" for item in overlap
    )
    proxy_beats_best = (
        proxy_metrics["attack_auc"]
        < best_metrics["attack_auc"] - 1.0e-12
    )
    proxy_verdict = (
        "SUPPORTED" if proxy_beats_best and not direct_leakage else "NOT_SUPPORTED"
    )
    claims = [
        {
            "claim": "Proxy-GAT adds value beyond the best traditional baseline.",
            "verdict": proxy_verdict,
            "evidence": {
                "proxy_attack_auc": proxy_metrics["attack_auc"],
                "best_baseline": best_baseline,
                "best_baseline_attack_auc": best_metrics["attack_auc"],
                "direct_label_feature_leakage": direct_leakage,
            },
        },
        {
            "claim": "The result generalizes to unseen graphs.",
            "verdict": "INCONCLUSIVE",
            "evidence": {
                "reason": (
                    "This workflow evaluates one graph transductively; it does not "
                    "supply a held-out graph or a frozen cross-graph model."
                )
            },
        },
        {
            "claim": "Proxy-GAT is merely a teacher reconstruction.",
            "verdict": (
                "SUPPORTED"
                if rank_spearman(rankings[proxy_name], rankings[teacher_name]) >= 0.9
                else "INCONCLUSIVE"
            ),
            "evidence": {
                "proxy_teacher_spearman": rank_spearman(
                    rankings[proxy_name], rankings[teacher_name]
                )
            },
        },
    ]
    return {
        "fractions": fraction_values.tolist(),
        "curves": {
            name: curve.largest_component_fractions.tolist()
            for name, curve in deterministic_curves.items()
        },
        "metrics": deterministic_metrics,
        "random": {
            "trials": random_trials,
            "mean_curve": random_mean_curve.largest_component_fractions.tolist(),
            "p05_curve": np.quantile(random_values, 0.05, axis=0).tolist(),
            "p95_curve": np.quantile(random_values, 0.95, axis=0).tolist(),
            "metrics": random_metrics,
        },
        "best_traditional_baseline": best_baseline,
        "proxy_beats_best_traditional": proxy_beats_best,
        "proxy_teacher_spearman": rank_spearman(
            rankings[proxy_name], rankings[teacher_name]
        ),
        "leakage_findings": overlap,
        "claims": claims,
    }


def _topology_only_target(
    graph: nx.Graph,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Measure each node's direct one-step largest-component loss.

    This target is an intervention on the supplied graph, not a weighted
    centrality score.  It keeps the later comparison to conventional
    centralities independent at the target-definition level.
    """

    node_count = graph.number_of_nodes()
    original_lcc = max(len(part) for part in nx.connected_components(graph))
    losses = np.zeros(node_count, dtype=float)
    for node in range(node_count):
        reduced = graph.copy()
        reduced.remove_node(node)
        remaining_lcc = (
            max(len(part) for part in nx.connected_components(reduced))
            if reduced.number_of_nodes()
            else 0
        )
        losses[node] = (original_lcc - remaining_lcc) / node_count
    return (
        torch.from_numpy(_minmax(losses).astype(np.float32)),
        {"single_node_lcc_loss": 1.0},
    )


def _sha256(path: Path) -> str:
    """Return a content hash for an input or generated artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plot_audit(evidence: Mapping[str, Any], output_path: Path) -> None:
    """Plot every deterministic strategy with a random uncertainty envelope."""

    fractions = np.asarray(evidence["fractions"], dtype=float)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    colors = {
        "proxy_gat": "#c0392b",
        "teacher": "#8e44ad",
        "pagerank": "#2471a3",
        "betweenness": "#117864",
        "degree": "#d68910",
        "closeness": "#566573",
        "eigenvector": "#7d3c98",
        "collective_influence": "#1f618d",
    }
    for name, values in evidence["curves"].items():
        axis.plot(
            fractions,
            np.asarray(values, dtype=float),
            label=name.replace("_", " ").title(),
            color=colors.get(name, "#2c3e50"),
            linewidth=2.1,
        )
    random_evidence = evidence["random"]
    lower = np.asarray(random_evidence["p05_curve"], dtype=float)
    upper = np.asarray(random_evidence["p95_curve"], dtype=float)
    mean = np.asarray(random_evidence["mean_curve"], dtype=float)
    axis.fill_between(fractions, lower, upper, color="#7f8c8d", alpha=0.18)
    axis.plot(fractions, mean, color="#626567", linewidth=2.2, label="Random mean")
    axis.axvspan(0.05, 0.10, color="#f5b7b1", alpha=0.22)
    axis.set(
        xlabel="Fraction of nodes removed",
        ylabel="Largest connected component / original nodes",
        title="CascadeRank Audit: node-ranking evidence",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=9)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, metadata={"Software": "CascadeRank Audit"})
    plt.close(figure)


def _markdown_report(evidence: Mapping[str, Any], graph_name: str) -> str:
    """Render fixed verdicts in a concise, human-readable evidence ledger."""

    leakage = evidence["leakage_findings"]
    lines = [
        "# CascadeRank Audit",
        "",
        f"Dataset: `{graph_name}`",
        "",
        "## Claim ledger",
        "",
    ]
    for item in evidence["claims"]:
        lines.extend(
            [
                f"### {item['verdict']}: {item['claim']}",
                "",
                "```json",
                json.dumps(item["evidence"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    lines.extend(["## Leakage audit", ""])
    if leakage:
        for item in leakage:
            lines.append(
                f"- **{item['severity'].upper()}** `{item['signal']}`: "
                f"{item['message']}"
            )
    else:
        lines.append("- No declared target-feature overlap was detected.")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This audit evaluates a ranking on one graph. It does not establish "
            "causal node importance or cross-graph generalization.",
            "",
        ]
    )
    return "\n".join(lines)


def _html_report(markdown: str, evidence: Mapping[str, Any]) -> str:
    """Create a lightweight judge-friendly static report without web services."""

    claim_rows = "".join(
        "<tr><td>{}</td><td class=\"{}\">{}</td></tr>".format(
            html.escape(item["claim"]),
            html.escape(item["verdict"].lower()),
            html.escape(item["verdict"]),
        )
        for item in evidence["claims"]
    )
    return """<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>CascadeRank Audit</title>
<style>
body{font-family:system-ui,sans-serif;max-width:980px;margin:32px auto;
line-height:1.5;color:#17202a}
table{border-collapse:collapse;width:100%}
td,th{border:1px solid #d5d8dc;padding:9px;text-align:left}
.supported{color:#196f3d;font-weight:700}.not_supported{color:#b03a2e;font-weight:700}.inconclusive{color:#7d6608;font-weight:700}
pre{background:#f4f6f7;padding:14px;overflow:auto}
img{max-width:100%;border:1px solid #d5d8dc}
</style></head>
<body><h1>CascadeRank Audit</h1><p>Machine-verified graph ranking claim ledger.</p>
<table><thead><tr><th>Claim</th><th>Verdict</th></tr></thead><tbody>""" + (
        claim_rows
    ) + """</tbody></table>
<h2>Attack evidence</h2><img src=\"attack_curves.png\" alt=\"Attack curves\">
<h2>Audit record</h2><pre>""" + html.escape(markdown) + (
        """</pre></body></html>"""
    )


def run_audit(config: AuditConfig) -> AuditRunResult:
    """Run the auditable ranking workflow and persist coherent artifacts."""

    if config.mode not in {"leaky", "topology-only"}:
        raise ValueError("mode must be 'leaky' or 'topology-only'")
    if not 1 <= config.epochs <= 50:
        raise ValueError("epochs must be between 1 and 50")
    if config.random_trials < 2:
        raise ValueError("random_trials must be at least two")
    start = time.perf_counter()
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_dataset(config.dataset, config.data_root, config.edge_csv)
    centralities = compute_centralities(loaded.graph, seed=config.seed)
    collective_influence = collective_influence_scores(loaded.graph)

    from cascaderank.pipeline import _build_proxy_training_data

    leaky_features, leaky_targets, proxy_metadata = _build_proxy_training_data(
        loaded, centralities
    )
    if config.mode == "leaky":
        features = leaky_features
        targets = leaky_targets
        target_signals = list(proxy_metadata["target_definition"].keys())
        feature_signals = [
            "degree",
            "betweenness",
            "closeness",
            "eigenvector",
            "pagerank",
            "clustering",
            "core",
            "neighbor-degree",
            "articulation",
            "brokerage",
        ]
        target_definition: Mapping[str, float] = proxy_metadata["target_definition"]
    else:
        features = torch.ones(
            (loaded.graph.number_of_nodes(), 1), dtype=torch.float32
        )
        targets, target_definition = _topology_only_target(loaded.graph)
        target_signals = list(target_definition)
        feature_signals = []

    proxy_scores, training = train_gat(
        features,
        loaded.edge_index,
        targets,
        epochs=config.epochs,
        seed=config.seed,
        hidden_channels=config.hidden_channels,
        device=config.device,
    )
    teacher_scores = {
        node: float(targets[node].item())
        for node in range(loaded.graph.number_of_nodes())
    }
    score_maps: dict[str, Mapping[int, float]] = {
        "proxy_gat": {
            node: float(proxy_scores[node])
            for node in range(loaded.graph.number_of_nodes())
        },
        "teacher": teacher_scores,
        **centralities,
        "collective_influence": collective_influence,
    }
    rankings = {
        name: stable_rank_nodes(scores) for name, scores in score_maps.items()
    }
    evidence = evaluate_ranking_audit(
        loaded.graph,
        rankings,
        target_signals=target_signals,
        feature_signals=feature_signals,
        random_trials=config.random_trials,
        seed=config.seed,
    )
    chart_path = output_dir / "attack_curves.png"
    _plot_audit(evidence, chart_path)
    markdown = _markdown_report(evidence, loaded.name)
    report_path = output_dir / "audit_report.md"
    report_path.write_text(markdown, encoding="utf-8")
    html_path = output_dir / "audit_report.html"
    html_path.write_text(_html_report(markdown, evidence), encoding="utf-8")
    input_hash = (
        _sha256(config.edge_csv.resolve())
        if config.edge_csv is not None
        else None
    )
    manifest = {
        "dataset": loaded.name,
        "nodes": loaded.graph.number_of_nodes(),
        "edges": loaded.graph.number_of_edges(),
        "mode": config.mode,
        "seed": config.seed,
        "epochs": config.epochs,
        "random_trials": config.random_trials,
        "target_definition": target_definition,
        "feature_signals": feature_signals,
        "input_sha256": input_hash,
        "artifact_sha256": {
            "attack_curves.png": _sha256(chart_path),
            "audit_report.md": _sha256(report_path),
            "audit_report.html": _sha256(html_path),
        },
        "training": training,
        "elapsed_seconds": time.perf_counter() - start,
        "evidence": evidence,
    }
    manifest_path = output_dir / "audit_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return AuditRunResult(
        manifest_path=manifest_path,
        report_path=report_path,
        html_path=html_path,
        chart_path=chart_path,
        evidence=evidence,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone audit command parser."""

    parser = argparse.ArgumentParser(
        prog="cascaderank-audit",
        description="Audit GNN critical-node ranking claims with fixed evidence.",
    )
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--edge-csv", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("audit_output"))
    parser.add_argument(
        "--mode", choices=("leaky", "topology-only"), default="leaky"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--hidden-channels", type=int, default=8)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit CLI and print only machine-readable output paths."""

    args = build_parser().parse_args(argv)
    try:
        result = run_audit(
            AuditConfig(
                dataset=args.dataset,
                edge_csv=args.edge_csv,
                data_root=args.data_root,
                output_dir=args.output_dir,
                mode=args.mode,
                epochs=args.epochs,
                seed=args.seed,
                random_trials=args.random_trials,
                hidden_channels=args.hidden_channels,
                device=args.device,
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"CascadeRank Audit failed: {exc}")
        return 1
    print(
        json.dumps(
            {
                "manifest": str(result.manifest_path),
                "report": str(result.report_path),
                "html": str(result.html_path),
                "chart": str(result.chart_path),
                "best_traditional_baseline": result.evidence[
                    "best_traditional_baseline"
                ],
                "claims": result.evidence["claims"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
