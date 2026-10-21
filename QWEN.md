# gliner — maintenance agent instructions

`gliner` is a Rust service that serves GLiNER bi-encoder models on ONNX Runtime (`ort`) behind an axum HTTP API.
Your job: when a dependency update (a Renovate branch) breaks CI, repair the code so it builds and passes again.
You edit files; the workflow commits. Anything you change outside the allowlist below is discarded automatically.

## Layout

| Path | Holds |
|---|---|
| `src/schema/` | Wire types and error enums: no logic, imports nothing else from the crate |
| `src/config/` | Configuration sections (serde) |
| `src/core/` | Pure logic: words splitter, windows, span decoding, label cache, tensors, settings, the `Recognizer` port; eval scorer, bootstrap, gate, example lines |
| `src/runtime/` | The artifact contract (`artifact.rs`), the ONNX Runtime backend (`onnx.rs`), the bounded executor; the model registry, data store, toolchain, train lab, job board and job |
| `src/api/` | axum handlers, the NDJSON batch stream, the server lifecycle, the train-role routes |
| `tests/api/`, `tests/server/` | Integration tests (fake recognizer, fake toolchain, real TCP lifecycle) |
| `tests/model/` | Model-backed parity, bench and train smoke, ignored in CI |
| `tests/resources/` | Fixtures (tiny ONNX artifact, PyTorch parity) — regenerated with `make tiny` / `make fixture`, never by hand |

Layers point inward: `schema ← config ← core ← runtime ← api` (`make bounds` enforces it).

## Allowlist

You may modify only: `src/`, `tests/api/`, `tests/server/`, `tests/model/`, `docs/`, `CHANGELOG.md`, `README.md`.
Never touch `.github/`, `Cargo.toml`, `Cargo.lock`, `tests/resources/`, `tools/`, `QWEN.md` or any version number.

## Rules

1. Keep the HTTP contract (routes, request and response shapes, status codes) unchanged. If an update forces a
   break, do not make it: describe it under "Needs a human decision".
2. Predictions must not change: offsets are characters, spans half-open, decoding greedy by score with stable ties,
   threshold strict. The model-free tests pin this; never weaken an assertion to make it pass.
3. No `unsafe`, no `unwrap()`/`expect()`/`panic!` outside tests, no indexing that can panic; `make lint` is clippy
   pedantic with warnings as errors.
4. Do not invent API. Use only what the pinned crate source shows (`~/.cargo/registry/src/*/<crate>-<version>/`).
5. Comments: none, except `// ##### SECTION #####` headers and a single `// ### reason` line where logic needs it;
   doc comments on public items. Tests are named `test_<target>_<scenario>`.
6. Add a line under `## [Unreleased]` in `CHANGELOG.md` for every user-visible change.

## Summary format

End with:

```
### Cause
<what the update changed and why CI failed>
### Fix
<files touched and why>
### Checks
<make check / make coverage results>
### Needs a human decision
<anything you did not do, or "nothing">
```
