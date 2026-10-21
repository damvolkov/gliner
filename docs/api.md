# API

Every error is `{"error": "<message>"}` with its status. Prediction routes answer `503` while the model loads. The
train routes ([below](#train-role)) answer `501` unless `train.enabled`.

## `POST /ner`

```bash
curl -s localhost:5800/ner -H 'content-type: application/json' \
  -d '{"text": "Linus Torvalds wrote Git in Helsinki.", "labels": ["person", "software tool", "location"]}'
```

```json
{"entities": [
  {"start": 0, "end": 14, "text": "Linus Torvalds", "label": "person", "score": 0.7146},
  {"start": 21, "end": 24, "text": "Git", "label": "software tool", "score": 0.7825},
  {"start": 28, "end": 36, "text": "Helsinki", "label": "location", "score": 0.6742}
]}
```

| Field | Type | |
|---|---|---|
| `text` | string | any length; windowed past the model's `max_len` words |
| `labels` | string[] | non-empty, none blank; free text |
| `threshold` | number? | in (0, 1); default `model.threshold` (0.5) |

`start..end` is a half-open range of characters. Entities never overlap and come ordered by start.

| Status | When |
|---|---|
| 400 / 415 | body is not JSON |
| 422 | unknown field, empty or blank label, threshold outside (0, 1) |
| 500 | inference failed |
| 503 | the model is still loading |

## `POST /ner/batch`

For thousands of texts on one connection. The request body is NDJSON, one item per line; the response is
`application/x-ndjson`, one line per item, **written as each item completes** (not in request order — match by `id`).

```bash
printf '%s\n' \
  '{"labels": ["person", "location"]}' \
  '{"id": "a", "text": "Ada Lovelace lived in London."}' \
  '{"id": "b", "text": "Linus Torvalds wrote Git.", "labels": ["person", "software tool"]}' \
| curl -sN localhost:5800/ner/batch -H 'content-type: application/x-ndjson' --data-binary @-
```

```json
{"id":"a","entities":[{"start":0,"end":12,"text":"Ada Lovelace","label":"person","score":0.7059},{"start":22,"end":28,"text":"London","label":"location","score":0.656}]}
{"id":"b","entities":[{"start":0,"end":14,"text":"Linus Torvalds","label":"person","score":0.7146},{"start":21,"end":24,"text":"Git","label":"software tool","score":0.7825}]}
```

| Line | Meaning |
|---|---|
| `{"labels"?, "threshold"?}` | **header**, optional, only before any item: defaults for every item after it |
| `{"id", "text", "labels"?, "threshold"?}` | **item**: its own labels/threshold win over the header's, the header's over the config's (`model.labels`, `model.threshold`) |
| blank | ignored |

| Reply | When |
|---|---|
| `{"id", "entities"}` | predicted |
| `{"id", "error"}` | the item is invalid (missing text, bad labels or threshold) or inference failed |
| `{"line", "error"}` | the line is not JSON, has unknown fields, lacks an `id`, is a late header, or exceeds `server.line` bytes |

A bad line never ends the stream. Flow control:

- at most `model.concurrency` predictions run at once across all requests; further items wait (never refused);
- the body is read no faster than items complete, and replies are produced no faster than the client reads them —
  backpressure both ways, constant memory;
- a client that disconnects cancels every pending item (a prediction already running finishes and is discarded);
- on shutdown (SIGTERM), open streams end after the items in flight.

## `GET /health`

`{"status": "ok", "version": "0.1.0"}` — liveness, always 200 once the port is bound.

## `GET /ready`

`200 {"status": "ready"}` once the model is loaded; `503 {"status": "loading"}` before.

## `GET /models`

```json
{"model": "knowledgator/gliner-bi-base-v2.0", "version": "0f6a855ea590f2ba", "labels_cached": 8, "served": 1}
```

`version` is the first 16 hex digits of the sha256 of the artifact files' sha256s: it changes with any byte of the
model, and with nothing else. `served` is the registry version answering (`current`).

## Train role

The flow, the registry and the gate: [Training](train.md).

| Route | Body → reply | Errors |
|---|---|---|
| `PUT /datasets/{name}` | NDJSON `{text, entities: [{start, end, label}]}` per line → `{name, lines, sha256}` | `422` bad name (1–64 of `A-Z a-z 0-9 . _ -`), bad or overlong line (`line N: …`), no examples |
| `PUT /evals/{name}` | the same, into the eval sets | the same |
| `GET /datasets`, `GET /evals` | → `[{name, lines, sha256}]` by name | |
| `POST /eval` | `{set, version?}` → `{set, version, texts, strict, lenient, labels: {label: score}}`, each score `{precision, recall, f1, tp, fp, fn}` | `404` set or version, `422` a set without labels |
| `POST /promote` | `{version}` → the new `/models` card | `404` version, `500` it does not load (the champion keeps serving) |
| `POST /train` | `{dataset, base?, params?: {epochs, lr, batch}, auto_promote?}` → `202` the job | `404` dataset or base, `422` params, `409` a job is running |
| `GET /train` | → every job of this process | |
| `GET /train/{id}` | → `{id, state, step, dataset, base, params, auto_promote, version, promoted, error, report, log, created, finished}` | `404` |
| `DELETE /train/{id}` | cancels; answers once stopped → the job, `state: "cancelled"` | `404`, `409` not running |

`state` is `running`, `succeeded`, `failed` or `cancelled`; `step` is `leakage`, `train`, `export`, `parity`,
`regression`, `gate`, `register` or `promote`; `log` keeps the last 200 lines.
