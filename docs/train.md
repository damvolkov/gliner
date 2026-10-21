# Training

Fine-tuning needs PyTorch; everything else stays Rust. The same binary runs in two roles:

| Role | Image | `train.enabled` | |
|---|---|---|---|
| serve | `docker/Dockerfile` (`make image`, 94 MB, distroless) | `false` | predictions only; every train route answers `501` |
| train | `docker/Dockerfile.train` (`make image-train`, 2.4 GB) | `true` | the same service, plus datasets, eval sets, evaluation, training, promotion |

The train image adds uv-managed Python with `tools/train` and `tools/export`. PyTorch is CPU by default;
`make image-train TORCH=cu128` swaps the training environment for a CUDA build (the CUDA runtime ships in the wheels;
the host needs the NVIDIA driver and `docker run --gpus all`). It runs as uid 65532 with the registry mounted
read-write at `/models` and the data dir at `/data` (datasets, eval sets, the Hugging Face cache):

```bash
make image-train
docker run -d --name gliner-train -p 5800:5800 \
  -v "$PWD/data/models/gliner-bi-base-v2.0:/models" -v "$PWD/data/train:/data" gliner-train:latest
```

## The model registry

`model.dir` is a registry: one directory per version, and a pointer to the one served.

```
data/models/gliner-bi-base-v2.0/
  current              "2" — the version served; only ever replaced whole (write, then rename)
  1/                   make export: the artifact (text.onnx, labels.onnx, tokenizers, config.json, manifest.json)
  2/                   a trained version: the artifact, plus
    checkpoint/        the PyTorch weights the next training continues from (train role only)
    manifest.json      the export's manifest, plus a `registry` block
```

```json
"registry": {
  "version": 2, "parent": 1, "base": "knowledgator/gliner-bi-base-v2.0",
  "dataset": {"name": "support", "lines": 1200, "sha256": "7bcb5b…"},
  "params": {"epochs": 3, "lr": 5e-06, "batch": 8},
  "created": 1761055912,
  "report": { …the job's report… }
}
```

`make export` writes version 1 (it has no `checkpoint/`: its training starts from the hub id in its manifest) and
points `current` at it. A version is registered by renaming a finished candidate directory into place as `max + 1`,
so it appears whole or not at all. Dot-directories (`.job-<id>`) are a running job's scratch space.

## Data in

Datasets (to train on) and eval sets (the internal regression base) share one format, NDJSON, one example per line:

```json
{"text": "Ana vive en Málaga", "entities": [{"start": 0, "end": 3, "label": "person"}, {"start": 12, "end": 18, "label": "location"}]}
```

Offsets are characters, half-open, inside the text; labels are free text, not blank; `"entities": []` is a valid
example. An upload is streamed and checked line by line, written to a temporary file and renamed into
`<train.data>/datasets/<name>.jsonl` or `<train.data>/evals/<name>.jsonl` only when every line holds: a set is replaced
whole or not at all. Eval sets live in the mounted volume, never in the repository; the repository ships one small
neutral set, `tests/resources/eval/neutral.jsonl`, for its tests.

```bash
curl -s -X PUT localhost:5800/evals/support --data-binary @support.eval.jsonl
# {"name":"support","lines":300,"sha256":"7843…"}
curl -s -X PUT localhost:5800/datasets/support --data-binary @support.train.jsonl
curl -s localhost:5800/evals
```

## Evaluation

`POST /eval {set, version?}` scores a registered version (default: the served one) on an eval set: each text is
predicted with the set's own labels (sorted) at `model.threshold`, then

- **strict**: a prediction counts when span and label match a gold entity;
- **lenient**: the distinct names a text mentions (casefolded, trimmed), label ignored;
- **per label**: strict, per gold or predicted label.

Precision, recall and F1 are micro-averaged over texts (counts summed, then divided; 0/0 is 0).

## A train job

`POST /train {dataset, base?, params?, auto_promote?}` answers `202` with the job; it then runs on its own. One job
at a time (`409`).

| Step | Does | Fails the job when |
|---|---|---|
| `leakage` | snapshots the dataset into the job's scratch dir (it trains on that copy, whatever `PUT` comes later) and fingerprints every text (casefolded, whitespace collapsed, sha256) | no eval set is registered, the dataset changed since the job started, or any eval text is a dataset text |
| `train` | `tools/train` from the base version's `checkpoint/`, or its hub id for version 1; the log streams into the job | the trainer fails |
| `export` | `tools/export` the checkpoint into the candidate dir, loaded by the Rust runtime | the export fails or does not load |
| `parity` | PyTorch's predictions on every eval text against the runtime's: Jaccard over `(text, start, end, label)` | parity < `train.parity` (0.99) |
| `regression` | candidate vs the champion (the version served) on every eval set and pooled, strict and lenient F1, each delta with a paired bootstrap 95 % interval (texts resampled, 1000 draws, fixed seed) | — |
| `gate` | on the pooled comparison: strict delta ≥ `gate.strict` (0), its interval's lower bound ≥ `gate.floor` (−1 point), lenient delta ≥ `gate.lenient` (−1 point) | — |
| `register` | the candidate becomes version `max + 1`, the full report in its manifest | — |
| `promote` | only with `auto_promote` and a passed gate; otherwise the version waits for `POST /promote` | — |

A candidate that fails the gate is still registered with its report, so it can be inspected and, by a human
decision, promoted. Whatever the outcome, the job's scratch dir is removed. `DELETE /train/{id}` kills the running
tool with everything it started (its process group) and answers once the job has stopped.

```bash
curl -s localhost:5800/train -H 'content-type: application/json' \
  -d '{"dataset": "support", "params": {"epochs": 3, "lr": 5e-6, "batch": 8}, "auto_promote": true}'
curl -s localhost:5800/train/1
```

The reply's shape (values illustrative):

```json
{"id": 1, "state": "succeeded", "step": "promote", "base": 1, "version": 2, "promoted": true, "error": null,
 "dataset": {"name": "support", "lines": 1200, "sha256": "7bcb5b…"}, "params": {"epochs": 3, "lr": 5e-06, "batch": 8},
 "auto_promote": true, "created": 1761055874, "finished": 1761055912,
 "report": {"champion": 1, "parity": 1.0,
   "sets": {"support": {"champion": {…}, "candidate": {…}, "strict": {…}, "lenient": {…}}},
   "pooled": {"champion": {…}, "candidate": {…},
              "strict": {"point": 2.4, "low": 0.3, "high": 4.6}, "lenient": {"point": 1.1, "low": -0.8, "high": 3.0}},
   "gate": {"passed": true, "checks": [{"name": "strict", "value": 2.4, "min": 0.0, "passed": true}, …]}},
 "log": ["step leakage", "eval sets checked: 1, no text shared", "step train", "{'loss': '17.25', …}", …, "succeeded"]}
```

Jobs live in memory: a restart forgets them, never what they registered.

## Promotion and hot reload

`POST /promote {version}` loads the version off the async runtime, rewrites `current`, then swaps the served model.
A request holds the model it started on to its end, so a swap never fails or cuts one short; a `/ner/batch` stream
runs on the model served when it opened. The swap keeps the same `model.concurrency` permits. A version that fails to
load returns an error and leaves the champion serving; `/models` reports the version served as `served`.
