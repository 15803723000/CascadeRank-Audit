# CascadeRank Audit: submission packet

## One-line description

CascadeRank Audit is a local Codex plugin that stops graph-ranking projects
from claiming a GNN advantage when leakage, weak baselines, or irreproducible
evidence do not support it.

## Problem

Critical-node ranking projects frequently train a GNN to reproduce centrality
signals and then compare the model with one of those same signals. A visually
convincing attack curve can therefore conceal direct label-feature leakage,
teacher reconstruction, or a missing uncertainty analysis. The scientific
claim and the available evidence are often mismatched.

## What the plugin does

The Codex skill runs two local experiments on the same graph: an intentionally
leaky counterexample and a repaired topology-only task. It produces a fixed
claim ledger, repeated random-ranking envelope, six non-learned baselines,
artifact hashes, and an independent verifier. It never upgrades a negative or
inconclusive result to a positive conclusion.

The workflow is local and requires no OpenAI API key. Codex is used through the
plugin skill to guide execution, inspect the machine-readable manifest before
the report, and explain the result boundary to the user.

## Demonstrable technical novelty

- Provenance audit: explicit label-feature and target-baseline overlap checks.
- Fixed evidence gate: a claim is `SUPPORTED` only under a predeclared AUC rule
  and only without direct leakage.
- Integrity gate: SHA-256 hashes cover each report artifact and the optional
  input CSV; `cascaderank.verify` independently recomputes verdict rules.
- Honest counterexample: the product shows that removing direct leakage does
  not automatically create a GNN advantage.

## Three-minute demo script

1. **0:00–0:20** — State the claim to audit: “This GNN finds more critical
   nodes than classical rankings.” Show `examples/bridge20/edges.csv`.
2. **0:20–0:55** — In Codex, invoke the CascadeRank Audit skill and run
   `scripts/run_demo.ps1`. Show that the leaky run flags direct overlap and
   reports `NOT_SUPPORTED` even though it was trained to reproduce the teacher.
3. **0:55–1:35** — Open the topology-only report. Explain that the target is
   observed single-node largest-component loss, with constant node inputs. The
   leakage flag disappears, but the GNN is still `NOT_SUPPORTED` because its
   AUC is worse than the selected baseline.
4. **1:35–2:10** — Run `python -m cascaderank.verify` on the manifest and edge
   CSV. Show `valid: true`, artifact hashes, and input-hash verification.
5. **2:10–2:40** — Change one character in a disposable report copy and rerun
   the verifier to show a hash mismatch. Do not modify the source repository.
6. **2:40–3:00** — State the product boundary: it audits one-graph claims; it
   does not infer causal importance or cross-graph generalization.

## Submission fields requiring real external values

- Public repository URL: `https://github.com/15803723000/CascadeRank-Audit`
- Public YouTube demo URL: `<PUBLIC_YOUTUBE_URL>`
- Codex `/feedback` Session ID: `<REAL_FEEDBACK_SESSION_ID>`
- License selected by the rights holder: `MIT`

Do not replace the remaining placeholders with invented values. The
implementation, public repository, and demo script are ready, but the public
video and feedback Session ID require actual account actions outside this
repository.
