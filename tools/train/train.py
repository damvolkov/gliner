"""Fine-tune a GLiNER bi-encoder on an NDJSON dataset and save the checkpoint the next training continues from.

dataset = one {"text", "entities": [{"start", "end", "label"}]} per line, character offsets, half-open: the
service's format. Words come from GLiNER's whitespace splitter; an entity becomes the run of words it overlaps.
The base is a checkpoint dir (a registered version's checkpoint/) or a Hugging Face id. Progress is printed per
step (transformers' PrinterCallback), the summary last, as JSON.
"""

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Final

import torch
from gliner import GLiNER

WORDS: Final = re.compile(r"\w+(?:[-_]\w+)*|\S")


class TrainRun:
    """One fine-tuning run: dataset in, checkpoint out."""

    def __init__(self, arguments: argparse.Namespace) -> None:
        self._args = arguments

    def _run_example(self, line: str) -> dict[str, object] | None:
        row = json.loads(line)
        words = [(found.start(), found.end(), found.group()) for found in WORDS.finditer(row["text"])]
        spans = [
            [index for index, (start, end, _) in enumerate(words) if start < entity["end"] and end > entity["start"]]
            for entity in row["entities"]
        ]
        ner = [
            [covered[0], covered[-1], entity["label"]]
            for covered, entity in zip(spans, row["entities"], strict=True)
            if covered
        ]
        return {"tokenized_text": [word for _, _, word in words], "ner": ner} if words else None

    def _run_examples(self) -> list[dict[str, object]]:
        with self._args.dataset.open(encoding="utf-8") as lines:
            examples = (self._run_example(line) for line in lines if line.strip())
            return [example for example in examples if example is not None]

    ############################################################

    def run(self) -> None:
        """The base loaded, fine-tuned on every example, saved whole (weights, config, both tokenizers)."""
        examples = self._run_examples()
        model = GLiNER.from_pretrained(self._args.base)
        cuda = torch.cuda.is_available()
        print(json.dumps({"examples": len(examples), "base": self._args.base, "cuda": cuda}), flush=True)
        with tempfile.TemporaryDirectory() as work:
            arguments = model.create_training_args(
                output_dir=work,
                learning_rate=self._args.lr,
                per_device_train_batch_size=self._args.batch,
                num_train_epochs=self._args.epochs,
                max_steps=-1,
                save_strategy="no",
                logging_steps=1,
                use_cpu=not cuda,
                dataloader_num_workers=0,
                disable_tqdm=True,
            )
            trainer = model.train_model(train_dataset=examples, eval_dataset=None, training_args=arguments)
        model.save_pretrained(self._args.out)
        summary = {"out": str(self._args.out), "examples": len(examples), "steps": trainer.state.global_step}
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="checkpoint dir or Hugging Face id to start from")
    parser.add_argument("--dataset", type=Path, required=True, help="NDJSON dataset")
    parser.add_argument("--out", type=Path, required=True, help="checkpoint dir to write")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch", type=int, default=8)
    TrainRun(parser.parse_args()).run()
