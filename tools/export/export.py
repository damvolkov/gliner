"""Export a GLiNER bi-encoder checkpoint to the artifact the Rust service serves.

model dir = text.onnx (text encoder + span scorer, label embeddings as an input), labels.onnx (label encoder,
mean-pooled and projected), tokenizer.json (text), labels.tokenizer.json (labels), config.json (the gliner config
subset the runtime reads), manifest.json (model id, revision, versions, sha256 of every file). The checkpoint is a
Hugging Face id or a local dir (a trained version's checkpoint/, exported under --name).
"""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Final

import numpy as np
import onnxruntime
import torch
from gliner import GLiNER
from huggingface_hub import HfApi
from torch import nn

OPSET: Final = 19
PROBE: Final = ["person", "organization", "machine learning model"]
TOLERANCE: Final = 1e-4
FILES: Final = ("text.onnx", "labels.onnx", "tokenizer.json", "labels.tokenizer.json", "config.json")
PACKAGES: Final = ("gliner", "torch", "transformers", "onnx", "onnxruntime")


class LabelsExport(nn.Module):
    """The bi-encoder's label tower alone: token ids → one pooled, projected embedding per label."""

    def __init__(self, model: GLiNER) -> None:
        super().__init__()
        self.encoder = model.model.token_rep_layer

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_labels(input_ids, attention_mask)


class ArtifactExport:
    """One checkpoint exported, verified against PyTorch and fingerprinted into a model dir."""

    def __init__(self, model_id: str, out: Path, name: str | None = None) -> None:
        self._id = model_id
        self._name = name or model_id
        self._out = out
        self._model = GLiNER.from_pretrained(model_id)
        self._model.model.to("cpu").eval()

    def _run_text(self) -> None:
        model = self._model
        ### gliner 0.2.29: export_to_onnx(from_labels_embeddings=True) forwards the flag to _build_dummy_batch and
        ### raises TypeError; the same internals, called in the right order, export the graph.
        batch = model._build_dummy_batch(labels=PROBE)
        inputs, spec = model._prepare_onnx_batch(batch, from_labels_embeddings=True, labels=PROBE)
        wrapper = model._create_onnx_wrapper(model.model).eval()
        ### The wrapper maps positional inputs by name, and the core calls the input `labels_embeds`, not the
        ### spec's `labels_embeddings`: a second break in the same path.
        wrapper.param_names = [name.replace("labels_embeddings", "labels_embeds") for name in spec["input_names"]]
        model._run_torch_onnx_export(
            wrapper,
            inputs,
            spec["input_names"],
            spec["output_names"],
            spec["dynamic_axes"],
            self._out / "text.onnx",
            OPSET,
        )

    def _run_labels(self) -> None:
        tokens = self._model.data_processor.labels_tokenizer(PROBE, return_tensors="pt", padding="longest")
        torch.onnx.export(
            LabelsExport(self._model).eval(),
            (tokens["input_ids"], tokens["attention_mask"]),
            f=str(self._out / "labels.onnx"),
            input_names=["input_ids", "attention_mask"],
            output_names=["embeddings"],
            dynamic_axes={
                "input_ids": {0: "labels", 1: "sequence_length"},
                "attention_mask": {0: "labels", 1: "sequence_length"},
                "embeddings": {0: "labels"},
            },
            opset_version=OPSET,
            dynamo=False,
        )

    def _run_files(self) -> None:
        processor = self._model.data_processor
        processor.transformer_tokenizer.backend_tokenizer.save(str(self._out / "tokenizer.json"))
        processor.labels_tokenizer.backend_tokenizer.save(str(self._out / "labels.tokenizer.json"))
        config = self._model.config
        subset = {
            "max_width": config.max_width,
            "max_len": config.max_len,
            "words_splitter_type": config.words_splitter_type,
            "subtoken_pooling": config.subtoken_pooling,
            "span_mode": config.span_mode,
            "labels_max_len": processor.labels_tokenizer.model_max_length,
        }
        (self._out / "config.json").write_text(json.dumps(subset, indent=2) + "\n")

    def _run_verify(self) -> float:
        tokens = self._model.data_processor.labels_tokenizer(PROBE, return_tensors="np", padding="longest")
        session = onnxruntime.InferenceSession(str(self._out / "labels.onnx"), providers=["CPUExecutionProvider"])
        exported = session.run(None, {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]})[0]
        with torch.no_grad():
            reference = self._model.encode_labels(PROBE).numpy()
        gap = float(np.abs(exported - reference).max())
        assert gap < TOLERANCE, f"labels.onnx drifts from encode_labels by {gap}"
        return gap

    def _run_digest(self, name: str) -> str:
        with (self._out / name).open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()

    def _run_manifest(self, gap: float) -> None:
        ### A local checkpoint (a trained version) has no hub revision.
        revision = None if Path(self._id).is_dir() else HfApi().model_info(self._id).sha
        manifest = {
            "model": self._name,
            "revision": revision,
            "opset": OPSET,
            "labels_gap": gap,
            "versions": {package: version(package) for package in PACKAGES},
            "files": {name: self._run_digest(name) for name in FILES},
        }
        (self._out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    ############################################################

    def run(self) -> None:
        """Every artifact file written, the label tower checked against PyTorch, the manifest last."""
        self._out.mkdir(parents=True, exist_ok=True)
        self._run_text()
        self._run_labels()
        self._run_files()
        self._run_manifest(self._run_verify())
        print(json.dumps({"out": str(self._out), "files": len(FILES) + 1}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face id or local dir of a GLiNER bi-encoder checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="model dir to write")
    parser.add_argument("--name", help="model id the manifest records (default: the model argument)")
    arguments = parser.parse_args()
    ArtifactExport(arguments.model, arguments.out, arguments.name).run()
