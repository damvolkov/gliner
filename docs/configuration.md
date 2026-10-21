# Configuration

One TOML file, `gliner.toml` in the working directory (or `--config PATH`), every key optional; every key can also be
set as `GLINER_<SECTION>__<KEY>` (e.g. `GLINER_MODEL__THREADS=8`), which wins over the file. Unknown keys and
out-of-range values are rejected at start with the key named; a configuration that cannot run never starts.

```toml
--8<-- "gliner.example.toml"
```

| Key | Default | |
|---|---|---|
| `server.host` | `0.0.0.0` | bind address |
| `server.port` | `5800` | |
| `server.line` | `1048576` | longest `/ner/batch` line, bytes (≥ 1024) |
| `server.log.level` | `info,ort=warn` | `EnvFilter` directive |
| `server.log.format` | `json` | `json` or `text` |
| `model.dir` | `data/models/gliner-bi-base-v2.0` | the artifact ([Model](model.md)) |
| `model.threads` | the CPUs this process may use | ONNX Runtime intra-op threads, always set explicitly: its own default sizes to the host, not the cgroup, and runs several times slower in a container |
| `model.concurrency` | `2` | predictions in flight across all requests |
| `model.threshold` | `0.5` | default score threshold, in (0, 1) |
| `model.labels` | `["person", "organization", "location"]` | default labels of `/ner/batch` items without their own |
| `model.cache` | `4096` | label embeddings kept |
| `model.window` | the model's `max_len` | words per window over long texts |
| `model.overlap` | `32` | words consecutive windows share (< window) |

Startup: the artifact is checked (files present, config supported, manifest complete) before the port is bound; the
sessions load in the background while `/health` answers and `/ready` says `loading`; a model that fails to load
(integrity, sessions) stops the process with a non-zero exit.
