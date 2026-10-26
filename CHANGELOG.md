# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

## [0.1.2] — 2026-10-26

Same model, same API: the project moves to the EDamLabs organisation.

### Changed

- Home: <https://github.com/edamlabs/gliner>, documentation <https://edamlabs.github.io/gliner/>; the old
  `damvolkov/gliner` URLs redirect. The image moves with the owner: `ghcr.io/edamlabs/gliner` from the next release.
- Identity declared in `Cargo.toml` (`[package.metadata.edamlabs]`: service, binding).
- Self-maintenance: routine Renovate updates (the tools' Python stack, images, toolchain, actions, lock files, the
  agent's CLI pin, Rust patch releases) merge themselves once every required check passes; the engine, the GLiNER
  library and Rust minor/major updates wait for a human. The agent reads the organisation's provider settings
  (`SM_*`), its commit gate checks renamed paths and no longer accepts edits under `tests/resources/`.
- Dependency health moves to esteward, the EDamLabs maintenance engine (daily: PEP 792 status, license changes, OSV
  advisories, action pins).

### Removed

- `health.yml`, `tools/health.py` and `make health`, covered by esteward.

### Security

- tokio 1.54.1: race condition in `tokio::time::Interval` (RUSTSEC-2026-0384, GHSA-5wjr-9356-4wc8), found by
  esteward on the migration.

## [0.1.1] — 2026-10-26

### Changed

- Settings load without `figment` (no release since 2024-05-17, flagged by the monthly health check): the file and
  the `GLINER_<SECTION>__<KEY>` overrides are TOML tables merged and deserialized with `toml` 1.1; an override's value
  is read as TOML (lists, numbers) or else as a string. Same contract, unknown keys still rejected. The settings tests
  no longer move the process's working directory, so the toolchain tests drop their `env -C /` workaround.

## [0.1.0] — 2026-10-21

### Added

- `POST /ner`, streaming `POST /ner/batch` (NDJSON), `GET /health`, `GET /ready`, `GET /models` over a GLiNER
  bi-encoder on ONNX Runtime: character offsets, windowing over long texts, cached label embeddings, flat greedy
  decoding on half-open spans.
- `tools/export`: a GLiNER bi-encoder checkpoint exported to the artifact contract (`text.onnx`, `labels.onnx`,
  tokenizers, config subset, manifest with sha256s), and the PyTorch parity fixture.
- Engine: ort 2.0.0-rc.15, tokenizers 0.23.2.
- Model registry: `model.dir` holds numbered versions and a `current` pointer (replaced atomically); the server
  serves `current`; `make export` writes version 1. `GET /models` reports the version `served`.
- The train role (`train.enabled`, the `gliner-train` image from `docker/Dockerfile.train`, `make image-train`,
  PyTorch CPU or `TORCH=cu128`): `PUT/GET /datasets`, `PUT/GET /evals` (NDJSON checked line by line, stored
  atomically), `POST /eval` (strict, lenient and per-label P/R/F1), `POST /promote` (hot swap; a failed load leaves
  the champion serving), `POST/GET /train`, `GET/DELETE /train/{id}`: one job at a time through leakage check,
  fine-tuning (`tools/train`), export, parity with PyTorch, regression against the champion with paired-bootstrap
  intervals, the gate (`[train.gate]`), registration, and promotion when asked and earned. Without the role these
  routes answer 501.
- `tools/train`; `tools/export` exports a local checkpoint (`--name`), `parity.py` takes `--labels`/`--threshold`.
- `make smoke`: one real CPU epoch through the whole job. CI builds the train image behind `TRAIN_IMAGE`.
- `make export INTO=<registry>`: a checkpoint exported as the next version of an existing registry, `current`
  untouched, so a candidate model is scored against the champion with `POST /eval`.
- Model watch: `tools/models.py` and a monthly `model-watch` issue listing GLiNER-family models newer than the served
  base (Apache-2.0/MIT), with whether this service can serve each; `make models`.
- Self-maintenance from the e-templates `selfmaintain` feature: `tools/health.py` (replaces `tools/health.sh`),
  `tools/upstream.py` (versions, release notes and public API of ort, tokenizers and gliner for the maintenance
  agent), lockfile refresh in `sync.yml`, `fix`/`expose` modes in `maintenance.yml`.

### Fixed

- A second model load in one process failed: ort allows a single environment per process, now shared by every load.
