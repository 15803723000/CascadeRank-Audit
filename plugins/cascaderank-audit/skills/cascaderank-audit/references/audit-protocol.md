# CascadeRank Audit Protocol

The reported outcome is a fixed-ranking node-removal benchmark. Lower attack
AUC means the largest connected component fell faster under that ranking.

The `leaky` mode is an intentional counterexample: centrality components build
both the target and model features. An apparent win in this mode is not
evidence that the GNN independently discovered critical nodes.

The `topology-only` mode uses observed single-node largest-component loss as
the target and constant node inputs. It tests a different, narrower prediction
task. It still cannot support a causal claim or a claim about a new graph until
the model and protocol are frozen and evaluated on held-out graphs.

Never change a verdict based on a plot's visual appearance. Use the manifest's
numeric evidence, leakage findings, and predeclared claim ledger.
