<h1 align="center">gliner</h1>

<p align="center">
  <strong>A GLiNER bi-encoder inference service in Rust: zero-shot named entities over HTTP, CPU-first, on ONNX Runtime,
  with the same predictions as the PyTorch reference.</strong>
</p>

<p align="center">
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.94-orange?logo=rust&logoColor=white" alt="Rust"></a>
  <a href="https://onnxruntime.ai/"><img src="https://img.shields.io/badge/runtime-ONNX%20Runtime%201.31-blue" alt="ONNX Runtime"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://github.com/j178/prek"><img src="https://img.shields.io/badge/hooks-prek-blueviolet" alt="prek"></a>
</p>

---

Labels are free text per request; the label tower encodes each once and caches it, the text tower scores every span
of up to 12 words against them. Offsets are **characters**, not bytes. Served: `knowledgator/gliner-bi-base-v2.0`
(ModernBERT text encoder, BGE-small label encoder), fp32 — the only precision that keeps PyTorch's predictions.

```
text ─▶ words ─▶ windows (max_len, overlap) ─▶ text.onnx ◀─ label embeddings ◀─ labels.onnx ◀─ LRU cache
                                                   │
                                     spans > threshold ─▶ greedy, half-open, non-overlapping ─▶ entities
```

| | |
|---|---|
| parity | 73/73 entities identical to PyTorch GLiNER 0.2.29 on the public fixture (Jaccard 1.0, max score drift 6.6e-7) |
| latency | 22.7 ms per ~100-character text, 108 ms per ~570-character text (8 threads, i9-14900K P-cores) |
| memory | 1.0 GB peak RSS; image 94 MB (distroless, model mounted) |

## Quickstart

```bash
make install                                             # pinned Rust, uv, prek, cargo-llvm-cov, cargo-about, hooks
make export MODEL=knowledgator/gliner-bi-base-v2.0       # PyTorch → data/models/gliner-bi-base-v2.0 (ONNX artifact)
make serve                                               # :5800
curl -s localhost:5800/ner -H 'content-type: application/json' \
  -d '{"text": "José Saramago vivió en Lanzarote.", "labels": ["person", "location"]}'
```

```json
{"entities":[{"start":0,"end":13,"text":"José Saramago","label":"person","score":0.70},
             {"start":23,"end":32,"text":"Lanzarote","label":"location","score":0.74}]}
```

Docker: `make image && make up` (the artifact dir is mounted read-only at `/models`; nothing is downloaded at run time).

## API

| Route | |
|---|---|
| `POST /ner` | `{text, labels[], threshold?}` → `{entities: [{start, end, text, label, score}]}` |
| `POST /ner/batch` | NDJSON in, NDJSON out, streamed: `{id, text, labels?, threshold?}` per line → `{id, entities}` or `{id, error}` as each completes |
| `GET /health` | liveness and version; answers while the model loads |
| `GET /ready` | 200 once the model is loaded, 503 before |
| `GET /models` | `{model, version, labels_cached}`; `version` hashes the model files |

Details, errors and the batch protocol: [docs/api.md](docs/api.md). Configuration: [docs/configuration.md](docs/configuration.md).
The artifact contract and the export: [docs/model.md](docs/model.md). Development, gates and workflows:
[docs/development.md](docs/development.md).

## License

MIT. ONNX Runtime (MIT) is linked statically; every Rust dependency is listed in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). Model weights keep their own licence (`knowledgator/gliner-bi-base-v2.0`: Apache-2.0).
