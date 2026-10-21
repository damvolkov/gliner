"""A tiny artifact with the real contract and arithmetic you can do by hand, for the service's model-free tests.

labels.onnx: embedding = 0.25 x (token count) on 4 dims, so a label's embedding sums to its token count.
text.onnx: logit(span, label) = span_mask x sum(label embedding) - 10 x span width. Every single word scores
sigmoid(token count of the label) on every label; wider spans fall far below any threshold.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Final

import onnx
from onnx import TensorProto, helper
from tokenizers import Tokenizer, models, pre_tokenizers, processors

OPSET: Final = 19
IR: Final = 9
WIDTH: Final = 12
DIMS: Final = 4
VOCAB: Final = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "person", "location", "organization", "tool", "software"]
CONFIG: Final = {
    "max_width": WIDTH,
    "max_len": 6,
    "words_splitter_type": "whitespace",
    "subtoken_pooling": "first",
    "span_mode": "markerV0",
    "labels_max_len": 8,
}
FILES: Final = ("text.onnx", "labels.onnx", "tokenizer.json", "labels.tokenizer.json", "config.json")


class TinyArtifact:
    """The tiny model dir, written file by file."""

    def __init__(self, out: Path) -> None:
        self._out = out

    def _run_text(self) -> onnx.ModelProto:
        node = helper.make_node
        inputs = [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "tokens"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "tokens"]),
            helper.make_tensor_value_info("words_mask", TensorProto.INT64, ["batch", "tokens"]),
            helper.make_tensor_value_info("text_lengths", TensorProto.INT64, ["batch", 1]),
            helper.make_tensor_value_info("span_idx", TensorProto.INT64, ["batch", "spans", 2]),
            helper.make_tensor_value_info("span_mask", TensorProto.BOOL, ["batch", "spans"]),
            helper.make_tensor_value_info("labels_embeddings", TensorProto.FLOAT, ["labels", DIMS]),
        ]
        output = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "words", WIDTH, "labels"])
        constants = [
            helper.make_tensor("zero", TensorProto.INT64, [], [0]),
            helper.make_tensor("one", TensorProto.INT64, [], [1]),
            helper.make_tensor("last", TensorProto.INT64, [1], [-1]),
            helper.make_tensor("axis", TensorProto.INT64, [1], [1]),
            helper.make_tensor("penalty", TensorProto.FLOAT, [], [10.0]),
            helper.make_tensor("middle", TensorProto.INT64, [2], [-1, WIDTH]),
        ]
        nodes = [
            node("Cast", ["span_mask"], ["mask"], to=TensorProto.FLOAT),
            node("Gather", ["span_idx", "zero"], ["start"], axis=2),
            node("Gather", ["span_idx", "one"], ["end"], axis=2),
            node("Sub", ["end", "start"], ["width_int"]),
            node("Cast", ["width_int"], ["width"], to=TensorProto.FLOAT),
            node("ReduceSum", ["labels_embeddings", "axis"], ["strength"], keepdims=0),
            node("Unsqueeze", ["mask", "last"], ["mask3"]),
            node("Mul", ["mask3", "strength"], ["base"]),
            node("Mul", ["width", "penalty"], ["cost"]),
            node("Unsqueeze", ["cost", "last"], ["cost3"]),
            node("Sub", ["base", "cost3"], ["flat"]),
            node("Shape", ["flat"], ["batch"], start=0, end=1),
            node("Shape", ["flat"], ["labels"], start=2, end=3),
            node("Concat", ["batch", "middle", "labels"], ["shape"], axis=0),
            node("Reshape", ["flat", "shape"], ["logits"]),
        ]
        graph = helper.make_graph(nodes, "tiny-text", inputs, [output], initializer=constants)
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], ir_version=IR)

    def _run_labels(self) -> onnx.ModelProto:
        node = helper.make_node
        inputs = [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["labels", "tokens"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["labels", "tokens"]),
        ]
        output = helper.make_tensor_value_info("embeddings", TensorProto.FLOAT, ["labels", DIMS])
        constants = [
            helper.make_tensor("axis", TensorProto.INT64, [1], [1]),
            helper.make_tensor("spread", TensorProto.FLOAT, [DIMS], [1 / DIMS] * DIMS),
        ]
        nodes = [
            node("Cast", ["attention_mask"], ["mask"], to=TensorProto.FLOAT),
            node("ReduceSum", ["mask", "axis"], ["count"], keepdims=1),
            node("Mul", ["count", "spread"], ["embeddings"]),
        ]
        graph = helper.make_graph(nodes, "tiny-labels", inputs, [output], initializer=constants)
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], ir_version=IR)

    def _run_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer(models.WordLevel({word: n for n, word in enumerate(VOCAB)}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
        )
        return tokenizer

    def _run_digest(self, name: str) -> str:
        with (self._out / name).open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()

    ############################################################

    def run(self) -> None:
        """Every file of the contract, the manifest last."""
        self._out.mkdir(parents=True, exist_ok=True)
        onnx.save(self._run_text(), self._out / "text.onnx")
        onnx.save(self._run_labels(), self._out / "labels.onnx")
        self._run_tokenizer().save(str(self._out / "tokenizer.json"))
        self._run_tokenizer().save(str(self._out / "labels.tokenizer.json"))
        (self._out / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
        manifest = {
            "model": "tiny",
            "revision": "tiny",
            "opset": OPSET,
            "versions": {"onnx": onnx.__version__},
            "files": {name: self._run_digest(name) for name in FILES},
        }
        (self._out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"out": str(self._out), "files": len(FILES) + 1}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="model dir to write")
    TinyArtifact(parser.parse_args().out).run()
