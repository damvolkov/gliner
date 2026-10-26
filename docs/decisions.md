# Decisions

### 2026-10-26 · The project moves to EDamLabs

**Decision.** `damvolkov/gliner` is transferred to `edamlabs/gliner`; documentation moves to
`edamlabs.github.io/gliner`, the image to `ghcr.io/edamlabs/gliner` from the next release.

**Evidence.** EDamLabs gathers the open-source bindings and services under one organisation, one GitHub App, one
set of provider settings (`SM_*`) and one maintenance engine (esteward); the release workflow already names the
image after the repository owner.

**Rejected.** Keeping the image under `damvolkov` (a second home for the same artefact).

### 2026-10-26 · Identity in Cargo.toml

**Decision.** `[package.metadata.edamlabs]` declares `kind = "service"` and `family = "binding"`; esteward's
`drift` module checks it against the fleet.

**Evidence.** The service ships a Docker image (no PyPI package) and ports a third-party model with its
predictions; a Rust project has no `pyproject.toml`, and Cargo ignores `metadata` tables.

**Rejected.** A `pyproject.toml` only for the declaration (a second manifest with no build role).

### 2026-10-26 · Routine updates merge themselves; health moves to esteward

**Decision.** Renovate merges the tools' Python stack, images, the toolchain, actions, lock files, the agent's CLI
pin and Rust patch releases after every required check; the engine (ort, tokenizers), the GLiNER library and Rust
minor or major updates wait for a human. `health.yml`, `tools/health.py` and `make health` are removed.

**Evidence.** CI carries the PyTorch parity fixture: a patch that changed predictions would fail it. esteward covers
what `health.yml` checked, plus PEP 792 status, license changes, OSV advisories and action pins.

**Rejected.** A human merging every green routine PR; two reports of the same facts.
