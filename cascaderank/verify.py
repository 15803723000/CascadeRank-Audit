"""Independent verifier for CascadeRank Audit evidence manifests.

The verifier deliberately does not trust the human-readable report.  It checks
the recorded hashes and recomputes the fixed verdict rules from the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REQUIRED_ARTIFACTS = (
    "attack_curves.png",
    "audit_report.md",
    "audit_report.html",
)
ADVANTAGE_CLAIM = "Proxy-GAT adds value beyond the best traditional baseline."
GENERALIZATION_CLAIM = "The result generalizes to unseen graphs."
RECONSTRUCTION_CLAIM = "Proxy-GAT is merely a teacher reconstruction."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _claims_by_text(claims: object) -> dict[str, Mapping[str, Any]]:
    if not isinstance(claims, list):
        return {}
    found: dict[str, Mapping[str, Any]] = {}
    for item in claims:
        if isinstance(item, dict) and isinstance(item.get("claim"), str):
            found[item["claim"]] = item
    return found


def verify_manifest(
    manifest_path: Path,
    edge_csv: Path | None = None,
) -> dict[str, Any]:
    """Verify integrity and deterministic claim rules for one audit manifest."""

    manifest_path = Path(manifest_path).resolve()
    errors: list[str] = []
    artifact_results: dict[str, bool] = {}
    input_hash_checked = False
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "valid": False,
            "manifest": str(manifest_path),
            "errors": [f"could not read manifest: {exc}"],
            "artifacts": artifact_results,
            "input_hash_checked": input_hash_checked,
        }
    if not isinstance(loaded, dict):
        errors.append("manifest root must be a JSON object")
        loaded = {}

    hashes = loaded.get("artifact_sha256")
    if not isinstance(hashes, dict):
        errors.append("artifact_sha256 is missing or invalid")
        hashes = {}
    for artifact in REQUIRED_ARTIFACTS:
        expected = hashes.get(artifact)
        artifact_path = manifest_path.parent / artifact
        if not isinstance(expected, str):
            errors.append(f"missing expected hash for {artifact}")
            artifact_results[artifact] = False
        elif not artifact_path.is_file():
            errors.append(f"missing artifact: {artifact}")
            artifact_results[artifact] = False
        else:
            artifact_results[artifact] = _sha256(artifact_path) == expected
            if not artifact_results[artifact]:
                errors.append(f"hash mismatch: {artifact}")

    expected_input_hash = loaded.get("input_sha256")
    if edge_csv is not None:
        input_hash_checked = True
        input_path = Path(edge_csv).resolve()
        if not input_path.is_file():
            errors.append(f"edge CSV does not exist: {input_path}")
        elif not isinstance(expected_input_hash, str):
            errors.append("manifest does not contain an input_sha256 value")
        elif _sha256(input_path) != expected_input_hash:
            errors.append("input edge CSV hash mismatch")

    evidence = loaded.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence is missing or invalid")
        evidence = {}
    claims = _claims_by_text(evidence.get("claims"))
    metrics = evidence.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("evidence.metrics is missing or invalid")
        metrics = {}
    proxy_metrics = metrics.get("proxy_gat")
    baseline_name = evidence.get("best_traditional_baseline")
    baseline_metrics = metrics.get(baseline_name)
    findings = evidence.get("leakage_findings")
    direct_leakage = isinstance(findings, list) and any(
        isinstance(item, dict)
        and item.get("kind") == "label_feature_overlap"
        for item in findings
    )
    if not isinstance(proxy_metrics, dict) or not isinstance(baseline_metrics, dict):
        errors.append("proxy or selected baseline metrics are missing")
    else:
        proxy_auc = proxy_metrics.get("attack_auc")
        baseline_auc = baseline_metrics.get("attack_auc")
        if not isinstance(proxy_auc, (int, float)) or not isinstance(
            baseline_auc, (int, float)
        ):
            errors.append("attack AUC values are missing or invalid")
        else:
            expected = (
                "SUPPORTED"
                if proxy_auc < baseline_auc - 1.0e-12 and not direct_leakage
                else "NOT_SUPPORTED"
            )
            actual = claims.get(ADVANTAGE_CLAIM, {}).get("verdict")
            if actual != expected:
                errors.append(
                    f"advantage verdict is {actual!r}; expected {expected!r}"
                )
    if claims.get(GENERALIZATION_CLAIM, {}).get("verdict") != "INCONCLUSIVE":
        errors.append("generalization verdict must be INCONCLUSIVE")
    reconstruction = claims.get(RECONSTRUCTION_CLAIM)
    spearman = evidence.get("proxy_teacher_spearman")
    if not isinstance(spearman, (int, float)):
        errors.append("proxy_teacher_spearman is missing or invalid")
    else:
        expected = "SUPPORTED" if spearman >= 0.9 else "INCONCLUSIVE"
        actual = reconstruction.get("verdict") if reconstruction else None
        if actual != expected:
            errors.append(
                f"reconstruction verdict is {actual!r}; expected {expected!r}"
            )
    random_evidence = evidence.get("random")
    trial_count_matches = isinstance(random_evidence, dict) and (
        random_evidence.get("trials") == loaded.get("random_trials")
    )
    if not trial_count_matches:
        errors.append("random trial count does not match the run configuration")

    return {
        "valid": not errors,
        "manifest": str(manifest_path),
        "errors": errors,
        "artifacts": artifact_results,
        "input_hash_checked": input_hash_checked,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cascaderank-verify",
        description="Verify CascadeRank Audit evidence and fixed verdict rules.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--edge-csv", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = verify_manifest(args.manifest, args.edge_csv)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
