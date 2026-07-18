# Local installation and support

CascadeRank Audit has no runtime OpenAI API requirement. The legacy natural
language report is optional and is not needed for the audit or verifier.

## Install the audit runtime

From this repository root:

```powershell
python -m pip install -e .
```

For tests and formatting checks:

```powershell
python -m pip install -e ".[dev]"
```

The optional legacy report dependency is installed only when explicitly needed:

```powershell
python -m pip install -e ".[legacy-report]"
```

## Load the local plugin

Load `plugins/cascaderank-audit` as a local plugin in Codex, then work in a
workspace where the CascadeRank Audit runtime is installed or where this source
repository is present. The plugin wrappers first search the active workspace
for `cascaderank/audit.py`; if no source tree is present, they invoke the
installed `cascaderank` package from the active Python environment.

## Smoke test

```powershell
.\scripts\run_demo.ps1
```

The command must produce two manifests and two successful verifier results. A
negative claim verdict is an expected product output, not a smoke-test failure.
