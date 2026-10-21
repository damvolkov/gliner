# Development

```bash
make install     # pinned Rust 1.94.0, uv, prek, cargo-llvm-cov, cargo-about, git hooks
make check       # lint (rustfmt + clippy pedantic, warnings are errors), bounds, gates, tests
make coverage    # ≥ 90 % lines, model-free (pre-push hook and CI)
make backends    # parity against PyTorch on the exported model (make export first)
make bench       # latency and peak RSS on the exported model
make help        # every target
```

## Layout

```
src/
  schema/    wire types and error enums                   (imports nothing of the crate)
  config/    configuration sections                       (schema-free)
  core/      words, windows, spans, label cache, tensors, settings, the Recognizer port
  runtime/   artifact contract, ONNX Runtime backend, bounded executor
  api/       axum handlers, NDJSON batch stream, server lifecycle
tests/
  api/       routes through tower::ServiceExt::oneshot over a fake recognizer (no network)
  server/    the real lifecycle over TCP: load, ready, predict, SIGTERM
  model/     parity and bench on the exported model (ignored by default)
  resources/ tiny/ (a 2 KB artifact with hand-checkable arithmetic), parity/ (texts + PyTorch predictions)
tools/
  export/    the uv project that exports a checkpoint (the only Python)
  health.sh  dependency health report
```

Layers point inward — `schema ← config ← core ← runtime ← api` — and `make bounds` fails on any import against
that direction. The model-free tests run the real ONNX Runtime path on `tests/resources/tiny` (regenerate with
`make tiny`), so runtime code is covered without the 770 MB model.

## Workflows

| Workflow | Trigger | Does | Gate |
|---|---|---|---|
| `ci.yml` | push to main, PRs | prek hooks, coverage, runtime image build; uploads the logs | always on |
| `docs.yml` | push to main | builds this site, deploys GitHub Pages | `DOCS_DEPLOY` |
| `release.yml` | manual | tags `v<Cargo.toml version>`, GitHub release; pushes `ghcr.io/<owner>/gliner` | image: `PUBLISH_IMAGE` |
| `health.yml` | monthly | `tools/health.sh`; opens/refreshes/closes a `deps-health` issue | always on (this repository only) |
| `sync.yml` | Renovate PRs | changelog line for an engine bump (ort, tokenizers), `THIRD_PARTY_LICENSES.md` | `RENOVATE_SYNC` + App |
| `maintenance.yml` | CI failed on a Renovate PR | an agent (`QWEN.md`) repairs within an allowlist; at most 3 attempts | `MAINTENANCE_AGENT` + App + `QWEN_API_KEY` |

`.github/renovate.json`: the engine (`ort`, `tokenizers`) one PR per release; the rest of the Rust tree, the export
tooling, images and actions grouped and scheduled; actions and images pinned by digest. Every variable a workflow
reads is declared inert in `.github/ci.vars.example`, and `make gates` fails otherwise.

## Releasing

Bump `version` in `Cargo.toml` and fold `## [Unreleased]` in `CHANGELOG.md` in a PR; after merge, `make release`.
