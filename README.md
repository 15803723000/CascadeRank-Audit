# CascadeRank Audit

CascadeRank Audit is an evidence-first, local workflow for auditing graph
critical-node ranking claims. It does not assume that a GNN is better than a
centrality baseline. It checks declared label/feature overlap, compares fixed
rankings under node-removal attacks, quantifies random-ranking uncertainty,
and emits a machine-readable claim ledger.

No OpenAI API key or network service is required for the audit command.

## What the audit can and cannot establish

The audit reports only three verdicts: `SUPPORTED`, `NOT_SUPPORTED`, and
`INCONCLUSIVE`. A result from one graph is never presented as causal node
importance or cross-graph generalization. The first claim is supported only if
the proxy has a lower attack AUC than the best traditional baseline and there
is no declared direct label-feature overlap.

`--mode leaky` deliberately reproduces the original proxy-target design. Its
target includes centrality signals also supplied to the GAT, so it should be
used as a teaching counterexample, not as a performance claim. `--mode
topology-only` instead supervises the GAT with each node's observed,
single-removal loss in the largest connected component. Its node features are
constant; message passing has access to topology only. This removes the direct
teacher-feature overlap and leaves conventional centralities as independent
baselines, but it still does not prove generalization.

The declared non-learned comparison set is Degree, Betweenness, Closeness,
Eigenvector, PageRank, and radius-2 Collective Influence. The manifest selects
the lowest-AUC member rather than assuming a preferred baseline.

## Install

Python 3.10–3.12 is supported.

```powershell
python -m pip install -r requirements.txt
```

For a minimal audit-only installation and local-plugin support, use
`python -m pip install -e .`; see [`INSTALLATION.md`](INSTALLATION.md).

## Run the two teaching cases

```powershell
python -m cascaderank.audit --dataset CSV --edge-csv .\examples\bridge20\edges.csv --mode leaky --random-trials 100 --output-dir .\audit_output\leaky
python -m cascaderank.audit --dataset CSV --edge-csv .\examples\bridge20\edges.csv --mode topology-only --random-trials 100 --output-dir .\audit_output\topology_only
```

For either Cora or PubMed, replace the dataset arguments with `--dataset Cora`
or `--dataset PubMed`. A custom edge CSV must have a header and either
`source,target` columns or two leading columns. Edges are treated as undirected;
self-loops are dropped and duplicate edges are merged.

## Audit artifacts

Each run writes the following coherent set to `--output-dir`:

- `audit_manifest.json`: configuration, target provenance, training metadata,
  numerical evidence, input hash, and hashes for all report artifacts.
- `audit_report.md` and `audit_report.html`: a fixed claim ledger and its
  interpretation boundary.
- `attack_curves.png`: every deterministic strategy plus the 5th–95th-percentile
  random-ranking envelope.

Verify the report before sharing it. This command fails if a report artifact,
the supplied input CSV, or a fixed verdict rule does not match the manifest:

```powershell
python -m cascaderank.verify --manifest .\audit_output\topology_only\audit_manifest.json --edge-csv .\examples\bridge20\edges.csv
```

For a local two-case demonstration (including verification after each run):

```powershell
.\scripts\run_demo.ps1
```

Attack AUC integrates the largest connected component divided by the original
node count; lower is a more effective fixed removal ranking. Checkpoints are
LCC at 5% and 10% removed. Rankings are fixed before deletion and are not
recomputed after each removal.

## Codex plugin

The local plugin bundle is under
[`plugins/cascaderank-audit`](plugins/cascaderank-audit). Its skill tells Codex
to read `audit_manifest.json` before interpreting results and prohibits
rewriting a negative or inconclusive verdict. It packages a no-key local audit
workflow for a Codex-based demonstration.

An English, evidence-constrained description and a three-minute demo sequence
are in [`SUBMISSION.md`](SUBMISSION.md). It intentionally leaves the public
repository URL, YouTube URL, feedback Session ID, and license as placeholders:
they must be filled with real submission artifacts.

## Built with Codex

Codex was used to construct and validate this project during the build period.
It helped implement the audit CLI, manifest verifier, Codex plugin wrappers,
test suite, reproducible demo, and release packaging. The decisive product and
research choices were to reject the original unsupported “GNN wins” narrative,
to expose the leaky proxy as a counterexample, and to preserve fixed negative
or inconclusive verdicts instead of optimizing for a favorable chart. The
repository history records these implementation stages and their validation
gates.

## Legacy pipeline

`main.py` remains for the original ranking demonstration, including its
optional OpenAI explanation mode. It is not the research-grade evidence path:
its original GAT target and structural inputs share centrality signals. Use
`python -m cascaderank.audit` for any defensible comparison or demo claim.

## Verify

```powershell
python -m pytest -q
python -m flake8 .
python -m compileall -q cascaderank main.py
python -m cascaderank.audit --help
```
