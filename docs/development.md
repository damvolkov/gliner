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
| `health.yml` | monthly | `tools/health.py` (crates.io, PyPI, GitHub); opens/refreshes/closes a `deps-health` issue | always on |
| `sync.yml` | Renovate PRs | lockfiles (`Cargo.lock`, the tools' `uv.lock`), changelog line for an engine bump (ort, tokenizers), `THIRD_PARTY_LICENSES.md` | `RENOVATE_SYNC` + App |
| `maintenance.yml` | CI done on a Renovate PR | an agent (`QWEN.md`) repairs a red CI (`fix`) or adopts new upstream API (`expose`) within an allowlist; at most 3 attempts | `MAINTENANCE_AGENT` + App + `QWEN_API_KEY` |

`.github/renovate.json`: the engine (`ort`, `tokenizers`; ort pins the ONNX Runtime it links) one PR per release, at
once; the `gliner` library its own PR; the rest of the Rust tree and of the tools' uv projects (torch, onnx,
onnxruntime, transformers, ruff, pytest) one weekly PR each; `rust-toolchain.toml`'s channel with the cargo-chef image
of the same minor; Dockerfile base images and actions pinned by digest, monthly. Every variable a workflow reads is
declared inert in `.github/ci.vars.example`, and `make gates` fails otherwise.

## Self-maintenance

Renovate opens the PRs, `sync.yml` keeps the lockfiles, changelog and licenses true, CI collects what changed
upstream, `maintenance.yml` mends the gaps, `health.yml` reports; a human merges. Nothing in this loop merges itself.

| Target | Files | Who | When |
|---|---|---|---|
| Dependency pins | `Cargo.toml`, `tools/*/pyproject.toml`, `docker/Dockerfile*`, `rust-toolchain.toml`, workflow digests | Renovate | a new release upstream |
| Lockfiles, licenses | `Cargo.lock`, `tools/**/uv.lock`, `THIRD_PARTY_LICENSES.md` | `sync.yml` | every Renovate push |
| Engine line | `CHANGELOG.md` → `## [Unreleased]` | `sync.yml` | an ort or tokenizers bump |
| Code, tests, tools, docs | `src/`, `tests/`, `tools/`, `docs/`, `CHANGELOG.md`, `README.md` | `maintenance.yml` agent | CI red, or the engine gained API |
| Health issue | issue labelled `deps-health` | `health.yml` | monthly |

Human-only: merging, the package version, release tags, `.github/`, `QWEN.md`, `Cargo.toml`, `prek.toml`. The
agent's limits are enforced by code: the workflow commits only allowlisted files and names every discarded one.

One-time setup (each step is one screen on github.com):

1. **GitHub App** — avatar → Settings → Developer settings → GitHub Apps → New. Webhook inactive. Repository
   permissions: Contents, Pull requests, Issues **read and write**; Metadata read-only; Workflows **no access** (the
   bots never rewrite `.github/`). Create, copy the Client ID, generate a private key (`.pem`), install it on this
   repository only.
2. **Hand it over** — `gh secret set APP_PRIVATE_KEY < key.pem` and `gh variable set APP_CLIENT_ID --body Iv1.…`.
3. **Renovate** — install https://github.com/apps/renovate on this repository only; it reads `.github/renovate.json`
   and opens the Dependency Dashboard issue.
4. **The agent** — `gh secret set QWEN_API_KEY`; optional variables `QWEN_MODEL`, `QWEN_BASE_URL`,
   `QWEN_CLI_VERSION` (`.github/ci.vars.example`).
5. **Labels** — `for l in upstream-update tooling ai-modified needs-human deps-health; do gh label create "$l" || true; done`.
6. **Gates** — `gh variable set RENOVATE_SYNC --body true`, then `MAINTENANCE_AGENT` once the loop is trusted.
7. **Protect `main`** — require a PR with one approval and the `ci` checks; no bypass, and the App off the allow list.

Verify: the Dependency Dashboard exists; a `renovate/*` PR gets a `chore(sync)` commit and an `upstream-context`
artifact; Actions → health → Run workflow opens or closes the `deps-health` issue. By hand: `make health`,
`uv run --no-project python -m tools.upstream --base <tree of main> --output context`.

## Releasing

Bump `version` in `Cargo.toml` and fold `## [Unreleased]` in `CHANGELOG.md` in a PR; after merge, `make release`.
