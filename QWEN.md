# QWEN.md — the agent contract for gliner

You are the maintenance agent of this repository, working on a Renovate branch after CI ran.
`gliner` is a Rust service that serves GLiNER bi-encoder models on ONNX Runtime (`ort`) behind an axum HTTP API;
two uv projects (`tools/export`, `tools/train`) export and fine-tune them in PyTorch.

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
| `tests/tools/` | Tests of the maintenance tools, on offline fixtures |
| `tests/resources/` | Fixtures (tiny ONNX artifact, PyTorch parity, Hub and lockfile replays) — regenerated with `make tiny` / `make fixture`, never by hand |
| `tools/export/`, `tools/train/` | The Python that exports and fine-tunes (gliner, torch, onnx, onnxruntime, transformers) |
| `tools/*.py` | The maintenance tools (health, upstream, models): stdlib only |
| `docs/` | The Zensical site. `README.md` — the front door. `CHANGELOG.md` — the record |

Layers point inward: `schema ← config ← core ← runtime ← api` (`make bounds` enforces it).

`.ai-context/` is what CI left you: `upstream-context/` (`versions.txt`, `release-notes.md`, `api-added.txt`,
`api-removed.txt`, `api-diff/<package>.diff` for ort, tokenizers and gliner) and `ci-logs/build-test.log` (why it
broke).

## Allowlist — you may edit ONLY these

`src/`, `tests/`, `tools/`, `docs/`, `CHANGELOG.md`, `README.md`.
Never `.github/`, `Cargo.toml`, `Cargo.lock`, `prek.toml`, `QWEN.md`, `tools/*/pyproject.toml`, `tools/*/uv.lock`,
`tests/resources/`, or any version number. The workflow discards anything outside the allowlist and says so on the PR.

## Rules

1. Fix is the job: make CI green with the smallest change the dependency update truly needs.
2. Expose is narrow: adopt new upstream API only where it replaces a workaround, a deprecated call or a hand-written
   equivalent. Never a new route, field, flag or setting.
3. Keep the HTTP contract (routes, request and response shapes, status codes) unchanged. If an update forces a break,
   do not make it: describe it under "Needs a human decision".
4. Predictions must not change: offsets are characters, spans half-open, decoding greedy by score with stable ties,
   threshold strict. The model-free tests pin this; never weaken an assertion to make it pass. An engine, gliner,
   torch or transformers bump needs parity re-checked by a human (`make backends`): say so.
5. No `unsafe`, no `unwrap()`/`expect()`/`panic!` outside tests, no indexing that can panic; `make lint` is clippy
   pedantic with warnings as errors. Do not invent API: use only what the pinned crate source shows
   (`~/.cargo/registry/src/*/<crate>-<version>/`) or `.ai-context/upstream-context/api-diff/`.
6. Comments: none, except `// ##### SECTION #####` headers and a single `// ### reason` line where logic needs it;
   doc comments on public items. Tests are named `test_<target>_<scenario>`. A behavior change gets a test first.
7. Every user-visible change gets one `CHANGELOG.md` line under `## [Unreleased]`.
8. Done means: `make check && make coverage`.
9. Do NOT commit or push — the workflow commits, through an allowlist gate, after you finish.

## Summary format (end your run with this)

```
### What changed upstream
<the update, and why CI failed or what API it gained>
### What I changed
<files touched and why>
### Tests added or modified
<names, or "none">
### Needs a human decision
<anything you did not do, parity to re-check, or "nothing">
```
