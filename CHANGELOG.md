# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

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

### Fixed

- A second model load in one process failed: ort allows a single environment per process, now shared by every load.
