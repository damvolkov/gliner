# gliner

A GLiNER bi-encoder inference service in Rust: zero-shot named entities over HTTP, CPU-first, on ONNX Runtime, with
the same predictions as the PyTorch reference.

An **EDamLabs binding**: a port of a third-party model that keeps its upstream's predictions and follows its releases, shipped as a service ([EDamLabs families](https://github.com/edamlabs/esteward#identity)).

## How it works

A bi-encoder has two towers. The **label tower** (BGE-small for `knowledgator/gliner-bi-base-v2.0`) turns each label
into one embedding; labels are free text per request, so each is encoded once and kept in an LRU cache. The **text
tower** (ModernBERT) encodes a window of words and scores every span of up to `max_width` (12) words against every
label embedding. Spans scoring strictly above the threshold are decoded greedily: by descending score, each kept
unless it shares a word with one already kept (half-open ranges: adjacent spans do not overlap), then ordered by start.

Texts longer than the model's `max_len` words are cut into overlapping windows; candidates from every window are
decoded together, so an entity seen by two windows is kept once.

Offsets are **characters** (Unicode scalar values) — what Python's `str` indexes and what a JSON client sees — never
bytes: `"José 🚀 Málaga"` puts `Málaga` at 7..13.

## Evidence

| | |
|---|---|
| parity | 73/73 entities identical to PyTorch GLiNER 0.2.29 over 24 public texts (Spanish and English, accents, emoji): Jaccard 1.0, identical lenient set, max score drift 6.6e-7 (`make backends`) |
| latency | 22.7 ms mean per ~100-character text (p95 31 ms), 108 ms per ~570-character text (p95 213 ms); 8 intra-op threads on i9-14900K P-cores (`make bench`) |
| memory | 1.0 GB peak RSS with the fp32 model loaded |
| image | 94 MB, distroless, non-root; the 770 MB artifact is mounted, not baked |

fp32 is the served precision: int8 quantization changes or drops predictions (Jaccard ≈ 0.39 against PyTorch in the
benchmark that chose this design). Batching texts hurts on CPU (padding), so texts are served one by one and
concurrency comes from in-flight predictions, bounded.
