---
name: cascaderank-audit
description: Run a local, evidence-first audit of a graph critical-node ranking or GNN claim. Use when a user supplies an edge CSV, asks whether a GNN improves on centrality baselines, needs a reproducible node-removal benchmark, or wants to demonstrate CascadeRank in Codex without an API key.
---

# CascadeRank Audit

Use this skill to test a claim; do not use it to manufacture a favorable GNN
narrative. Read `references/audit-protocol.md` before interpreting a result.

## Workflow

1. Identify the graph input and write the precise claim under test. Require an
   undirected edge CSV with `source,target` headers when the user supplies data.
2. Run the deliberately leaky counterexample and the repaired topology-only
   audit on the same graph. From the repository root:

   ```powershell
   python plugins/cascaderank-audit/skills/cascaderank-audit/scripts/run_audit.py --dataset CSV --edge-csv <edges.csv> --mode leaky --random-trials 100 --output-dir audit_output/leaky
   python plugins/cascaderank-audit/skills/cascaderank-audit/scripts/run_audit.py --dataset CSV --edge-csv <edges.csv> --mode topology-only --random-trials 100 --output-dir audit_output/topology_only
   ```

3. Read `audit_manifest.json` before the human-readable report. Check input and
   artifact hashes, `leakage_findings`, all `claims`, the best baseline, and
   random-trial count. Run the independent verifier before reporting:

   ```powershell
   python plugins/cascaderank-audit/skills/cascaderank-audit/scripts/verify_audit.py --manifest audit_output/topology_only/audit_manifest.json --edge-csv <edges.csv>
   ```
4. Report the verdict exactly. `NOT_SUPPORTED` means the claimed advantage was
   not established. `INCONCLUSIVE` means the experiment did not test the
   broader claim. Do not convert either into a positive conclusion.
5. State the scope boundary: one-graph transductive results do not establish
   causal node importance or cross-graph generalization. Propose held-out graph
   tests only as a next experiment, not as a result already obtained.

## Output contract

Return the audit directory and a compact evidence table containing the proxy
attack AUC, best traditional attack AUC, direct leakage flag, random trial
count, and all three claim verdicts. Link the static HTML report when available.
