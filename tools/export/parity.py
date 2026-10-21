"""The parity fixture: PyTorch GLiNER predictions over a set of texts, the reference the Rust service must match.

Input: a JSON list of {"id", "text"}. Output: {model, labels, threshold, items: [{id, text, entities}]}, entities as
the service returns them (character offsets, half-open). Labels and threshold default to the public fixture's; the
train role passes an eval set's own.
"""

import argparse
import json
from pathlib import Path
from typing import Final

from gliner import GLiNER

LABELS: Final = [
    "service",
    "software library",
    "software tool",
    "protocol or standard",
    "machine learning model",
    "person",
    "organization",
    "location",
]
THRESHOLD: Final = 0.5


class ParityFixture:
    """Reference predictions of one checkpoint, written as a fixture."""

    def __init__(self, model_id: str, labels: list[str], threshold: float) -> None:
        self._id = model_id
        self._labels = labels
        self._threshold = threshold
        self._model = GLiNER.from_pretrained(model_id)
        self._model.model.to("cpu").eval()

    def _run_item(self, row: dict[str, str]) -> dict[str, object]:
        found = self._model.predict_entities(row["text"], self._labels, threshold=self._threshold)
        entities = [
            {"start": e["start"], "end": e["end"], "text": e["text"], "label": e["label"], "score": e["score"]}
            for e in found
        ]
        return {"id": row["id"], "text": row["text"], "entities": entities}

    ############################################################

    def run(self, texts: Path, out: Path) -> None:
        """Every text predicted with the labels and threshold, written to `out`."""
        items = [self._run_item(row) for row in json.loads(texts.read_text())]
        fixture = {"model": self._id, "labels": self._labels, "threshold": self._threshold, "items": items}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n")
        print(json.dumps({"out": str(out), "items": len(items), "entities": sum(len(i["entities"]) for i in items)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face id or local dir of the checkpoint the artifact was exported from")
    parser.add_argument("--texts", type=Path, required=True, help='JSON list of {"id", "text"}')
    parser.add_argument("--out", type=Path, required=True, help="fixture to write")
    parser.add_argument("--labels", type=json.loads, default=LABELS, help="JSON list of labels")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    arguments = parser.parse_args()
    ParityFixture(arguments.model, arguments.labels, arguments.threshold).run(arguments.texts, arguments.out)
