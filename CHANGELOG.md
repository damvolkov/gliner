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
