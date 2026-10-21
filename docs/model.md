# Model

## The artifact contract

The service loads a directory, never a hub id, and downloads nothing; `model.dir` is a registry of such
directories, one per version ([Training](train.md#the-model-registry)), `current` naming the one served:

| File | |
|---|---|
| `text.onnx` | text tower + span scorer; inputs `input_ids`, `attention_mask`, `words_mask`, `text_lengths`, `span_idx`, `span_mask`, `labels_embeddings`; output `logits [batch, words, max_width, labels]` |
| `labels.onnx` | label tower, mean-pooled and projected; inputs `input_ids`, `attention_mask`; output `embeddings [labels, dim]` |
| `tokenizer.json` | the text tokenizer |
| `labels.tokenizer.json` | the label tokenizer (a bi-encoder has two) |
| `config.json` | the GLiNER config subset the runtime reads: `max_width`, `max_len`, `words_splitter_type` (`whitespace`), `subtoken_pooling` (`first`), `span_mode`, `labels_max_len` |
| `manifest.json` | model id, revision, opset, package versions, sha256 of every file above |

At load every file is hashed against the manifest; a mismatch refuses to serve.

## Export

```bash
make export MODEL=knowledgator/gliner-bi-base-v2.0     # → data/models/gliner-bi-base-v2.0/1, current = 1
```

It refuses to overwrite an existing version 1. `INTO=<registry>` exports into an existing registry as its next version
instead, staged in a dot-dir and renamed into place, `current` untouched:

```bash
make export MODEL=knowledgator/gliner-bi-large-v2.0 INTO=gliner-bi-base-v2.0   # → data/models/gliner-bi-base-v2.0/2
```

A trained version is exported by the train job from its local checkpoint (`export.py <checkpoint dir> --name <lineage id>`).

`tools/export` is a uv project (PyTorch CPU, gliner 0.2.29) — with `tools/train`, the only Python here, because
exporting from and training in PyTorch have no Rust path. It works around two bugs of gliner 0.2.29's `export_to_onnx(from_labels_embeddings=True)`: the
flag reaches `_build_dummy_batch` (TypeError), and the wrapper names the embeddings input `labels_embeddings` where
the core expects `labels_embeds`. The label tower is exported separately and checked against `encode_labels`
(max gap ≈ 2e-7) before the manifest is written.

Any GLiNER **bi-encoder** whose config uses the `whitespace` splitter and `first` sub-token pooling fits the
contract; uni-encoders (labels in the prompt) do not.

## Model watch

`.github/workflows/models.yml` (monthly, and on demand) runs `tools/models.py` over `tools/models.toml`: the served
base, the licences accepted (Apache-2.0, MIT), the tracked Hub searches (knowledgator `gliner-bi*`/`gliner-x*`,
urchade `gliner*`, fastino `gliner2*`, numind `NuNER*`, and every model tagged `gliner` above a download floor) and
the ids already triaged (`ignore`, globs allowed). It opens or refreshes one issue labelled `model-watch` listing every
candidate created after the base — id, date, licence, downloads, parameters when the Hub knows them, its architecture
from `gliner_config.json`, and whether this service can serve it — and closes it when nothing is newer. Servable
candidates come first. `make models` prints the same report locally.

The issue is triaged by a human; no workflow and no agent ever swaps the served model.

### Evaluating a candidate

Only a candidate marked servable (a bi-encoder in span mode, whitespace words, first sub-token pooling) fits the
artifact contract. Evaluate it with the service itself, against the eval sets registered on the train role — the
internal regression base the train job's gate uses:

```bash
make export MODEL=<candidate id> INTO=gliner-bi-base-v2.0       # → the registry's next version, say 3
curl -s localhost:5800/evals                                      # the registered eval sets
curl -s localhost:5800/eval -H 'content-type: application/json' -d '{"set": "support"}'               # champion
curl -s localhost:5800/eval -H 'content-type: application/json' -d '{"set": "support", "version": 3}' # candidate
```

Compare strict, lenient and per-label F1 set by set; then latency and memory on a registry of its own
(`make export MODEL=<candidate id>`, then `GLINER_MODEL__DIR=data/models/<name> make bench`), and the licence of its
weights. Promote it
(`POST /promote {"version": 3}`) only by that decision; a candidate that loses stays registered, unserved, or is
deleted by hand. Then set `base` in `tools/models.toml` and `MODEL` in the Makefile to the new base.

## Parity

`make fixture` runs PyTorch GLiNER over `tests/resources/parity/texts.json` (24 neutral texts) and writes the
reference predictions; `make backends` runs the exported model over the same texts and requires ≥ 99 % identical
predictions (Jaccard over `(text, start, end, label)`) and an identical lenient set (`(text, casefolded entity)`).
Any other fixture in the same format can be checked with `PARITY_FIXTURE=path make backends`.

## Benchmark

`make bench` reports ms per text and peak RSS over the fixture texts; pin threads with `GLINER_MODEL__THREADS`.
