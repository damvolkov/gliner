# Development

```bash
make install     # pinned Rust 1.94.0, uv, prek, cargo-llvm-cov, cargo-about, git hooks
make check       # lint (rustfmt + clippy pedantic, warnings are errors), bounds, gates, tests
make coverage    # ≥ 90 % lines, model-free (pre-push hook and CI)
make backends    # parity against PyTorch on the exported model (make export first)
make bench       # latency and peak RSS on the exported model
make smoke       # one real CPU epoch through the whole train job (make export first)
make image-train # the train image [TORCH=cpu|cu128]
make help        # every target
```

## Layout

```
src/
  schema/    wire types and error enums                   (imports nothing of the crate)
  config/    configuration sections                       (schema-free)
  core/      words, windows, spans, label cache, tensors, settings, the Recognizer port;
             examples and leakage, scorer, paired bootstrap, regression gate
  runtime/   artifact contract, ONNX Runtime backend, bounded executor and its swappable slot; model registry,
             data store, the toolchain port and its process adapter, the train lab, job board and job
  api/       axum handlers, NDJSON batch stream, server lifecycle, train-role routes
tests/
  api/       routes through tower::ServiceExt::oneshot over a fake recognizer, and the train role over the tiny
             artifact with a fake toolchain (no network, no PyTorch)
  server/    the real lifecycle over TCP: load, ready, predict, train role on, SIGTERM
  model/     parity, bench and the train smoke on the exported model (ignored by default)
  tools/     the maintenance tools' tests (pytest), on offline replays in resources/
  resources/ tiny/ (a 2 KB artifact with hand-checkable arithmetic), parity/ (texts + PyTorch predictions),
             eval/ and train/ (small neutral NDJSON sets)
tools/
  export/    the uv project that exports a checkpoint
  train/     the uv project that fine-tunes one
  health.py      dependency health report (crates.io, PyPI, GitHub)      stdlib only:
  upstream.py    what a Renovate PR changed: versions, notes, public API  python -m tools.<name>
  models.py      model watch over models.toml (the Hugging Face Hub)
  pyproject.toml their tests (tests/tools, offline fixtures) and ruff: make pytest
```

Layers point inward — `schema ← config ← core ← runtime ← api` — and `make bounds` fails on any import against
that direction. The model-free tests run the real ONNX Runtime path on `tests/resources/tiny` (regenerate with
`make tiny`), so runtime code is covered without the 770 MB model.

## Workflows

| Workflow | Trigger | Does | Gate |
|---|---|---|---|
| `ci.yml` | push to main, PRs | prek hooks, coverage, runtime image build; uploads the logs; on `renovate/*` PRs, `upstream-context` (`tools/upstream.py`: versions, release notes, public API diff of ort, tokenizers and gliner) | always on; train image: `TRAIN_IMAGE` |
| `docs.yml` | push to main | builds this site, deploys GitHub Pages | `DOCS_DEPLOY` |
| `release.yml` | manual | tags `v<Cargo.toml version>`, GitHub release; pushes `ghcr.io/<owner>/gliner:X.Y.Z` and `:latest` | image: `PUBLISH_IMAGE` |
| `models.yml` | monthly | `tools/models.py`: GLiNER-family models on the Hub newer than the served base; opens/refreshes/closes a `model-watch` issue ([Model](model.md#model-watch)) | always on |
| `sync.yml` | Renovate PRs | lockfiles (`Cargo.lock`, the tools' `uv.lock`), changelog line for an engine bump (ort, tokenizers), `THIRD_PARTY_LICENSES.md` | `RENOVATE_SYNC` + App |
| `maintenance.yml` | CI done on a Renovate PR | an agent (`QWEN.md`) repairs a red CI (`fix`) or adopts new upstream API (`expose`) within an allowlist; at most 3 attempts | `MAINTENANCE_AGENT` + App + `SM_API_KEY` |

`.github/renovate.json`: the engine (`ort`, `tokenizers`; ort pins the ONNX Runtime it links) one PR per release, at
once; the `gliner` library its own PR; the rest of the Rust tree and of the tools' uv projects (torch, onnx,
onnxruntime, transformers, ruff, pytest) one weekly PR each; `rust-toolchain.toml`'s channel with the cargo-chef image
of the same minor; Dockerfile base images and actions pinned by digest, monthly. Every variable a workflow reads is
declared inert in `.github/ci.vars.example`, and `make gates` fails otherwise.

## Self-maintenance

Renovate opens the PRs, `sync.yml` keeps the lockfiles, changelog and licenses true, CI collects what changed
upstream, `maintenance.yml` mends the gaps. Routine updates merge themselves once every required check passes
(the tools' Python stack, images, the toolchain, actions, lock files, the agent's CLI pin, Rust patch releases);
the engine (ort, tokenizers), the GLiNER library and every Rust minor or major wait for a human. Dependency health
and security are watched from outside by [esteward](https://github.com/edamlabs/esteward)
([`fleet/gliner.toml`](https://github.com/edamlabs/esteward/blob/main/fleet/gliner.toml)), daily.

| Target | Files | Who | When |
|---|---|---|---|
| Dependency pins | `Cargo.toml`, `tools/*/pyproject.toml`, `docker/Dockerfile*`, `rust-toolchain.toml`, workflow digests | Renovate | a new release upstream |
| Lockfiles, licenses | `Cargo.lock`, `tools/**/uv.lock`, `THIRD_PARTY_LICENSES.md` | `sync.yml` | every Renovate push |
| Engine line | `CHANGELOG.md` → `## [Unreleased]` | `sync.yml` | an ort or tokenizers bump |
| Code, tests, tools, docs | `src/`, `tests/` (not `tests/resources/`), `tools/`, `docs/`, `CHANGELOG.md`, `README.md` | `maintenance.yml` agent | CI red, or the engine gained API |
| Health and security | issues labelled `esteward` (status) and `esteward-escalation` (one per incident) | esteward | daily |
| Model watch | issue labelled `model-watch` (human triage only) | `models.yml` | monthly |

Human-only: merging the engine and minor/major updates, the package version, release tags, `.github/`, `QWEN.md`, `Cargo.toml`, `prek.toml`. The
agent's limits are enforced by code: the workflow commits only allowlisted files and names every discarded one.

One-time setup: the EDamLabs organisation provides the GitHub App (`eself-maintenance`, installed on every
repository; organisation variable `APP_CLIENT_ID`, secret `APP_PRIVATE_KEY`), the agent's provider (secret
`SM_API_KEY`, variables `SM_BASE_URL`, `SM_MODEL`), Renovate (Mend, repository selected) and esteward. Per
repository only:

1. **Labels** — `for l in upstream-update tooling ai-modified needs-human model-watch; do gh label create "$l" || true; done`.
2. **Gates** — `gh variable set RENOVATE_SYNC --body true`, then `MAINTENANCE_AGENT` once the loop is trusted
   (`.github/ci.vars.example`).
3. **Protect `main`** — a PR and the `ci` checks, strict; no required approval (routine updates merge themselves),
   no bypass for the App; auto-merge allowed on the repository.

Verify: the Dependency Dashboard exists; a `renovate/*` PR gets a `chore(sync)` commit and an `upstream-context`
artifact; Actions → models → Run workflow opens or closes the `model-watch` issue; esteward's status issue
(label `esteward`) is rewritten daily. By hand: `make models`,
`uv run --no-project python -m tools.upstream --base <tree of main> --output context`.

## Releasing

Bump `version` in `Cargo.toml` and fold `## [Unreleased]` in `CHANGELOG.md` in a PR; after merge, `make release`.
