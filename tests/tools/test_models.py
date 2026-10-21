from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from tools import models


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            {"labels_encoder": "BAAI/bge-small-en-v1.5", "span_mode": "markerV0"},
            ("bi-encoder · span (markerV0)", True),
        ),
        ({"labels_encoder": None, "span_mode": "markerV0"}, ("uni-encoder · span (markerV0)", False)),
        ({"labels_encoder": "x", "span_mode": "token_level"}, ("bi-encoder · token", False)),
        (
            {"labels_encoder": "x", "words_splitter_type": "spacy", "subtoken_pooling": "mean"},
            ("bi-encoder · span (markerV0) · spacy/mean", False),
        ),
        (None, ("unknown: no gliner_config.json", False)),
    ],
    ids=["bi-span", "uni", "token", "splitter", "missing"],
)
def test_architecture_support(config: dict[str, Any] | None, expected: tuple[str, bool]) -> None:
    assert models.architecture(config) == expected


def test_search_url_query() -> None:
    query = parse_qs(urlsplit(models.search_url({"author": "acme", "min_downloads": 5})).query)
    assert (query["author"], query["sort"], "min_downloads" in query) == (["acme"], ["createdAt"], False)
    assert {"createdAt", "downloads", "cardData", "safetensors"} <= set(query["expand[]"])


def test_candidates_filtered_and_classified(hub_config: Path, hub: Callable[[str], Any]) -> None:
    found = models.candidates(models.load(hub_config), hub)
    assert [(c.id, c.license, c.supported) for c in found] == [
        ("acme/gliner-bi-large", "apache-2.0", True),
        ("other/gliner-token", "mit", False),
        ("other/gliner-uni", "mit", False),
        ("other/gliner2", "apache-2.0", False),
    ]
    assert (found[0].params, found[0].created, found[-1].architecture) == (
        530000000,
        "2025-09-01",
        "unknown: no gliner_config.json",
    )


def test_report_table(hub_config: Path, hub: Callable[[str], Any]) -> None:
    lines = models.report(models.candidates(models.load(hub_config), hub))
    assert lines[0] == "4 candidates, 1 servable here (bi-encoder, span mode, whitespace words, first sub-token)."
    assert lines[4].endswith("| 2025-09-01 | apache-2.0 | 50 | 530M | bi-encoder · span (markerV0) | yes |")
    assert lines[5] == (
        "| [other/gliner-token](https://huggingface.co/other/gliner-token) | 2026-03-01 | mit | 3000 | ? "
        "| bi-encoder · token | no |"
    )


@pytest.mark.parametrize(
    ("ignore", "expected"), [((), models.FINDINGS_EXIT), (("*",), 0)], ids=["candidates", "nothing-new"]
)
def test_run_exit(
    hub_config: Path, hub: Callable[[str], Any], tmp_path: Path, ignore: tuple[str, ...], expected: int
) -> None:
    config = tmp_path / "models.toml"
    config.write_text(hub_config.read_text().replace('ignore = ["noise/*"]', f"ignore = {list(ignore)!r}"))
    output = tmp_path / "report.md"
    assert (models.run(config, output, hub), output.read_text().count("| [")) == (
        expected,
        5 if ignore == () else 0,
    )


def test_candidates_base_missing(hub_config: Path, hub: Callable[[str], Any], tmp_path: Path) -> None:
    config = tmp_path / "models.toml"
    config.write_text(hub_config.read_text().replace('base = "acme/gliner-bi-base"', 'base = "acme/gone"'))
    with pytest.raises(LookupError, match="acme/gone"):
        models.candidates(models.load(config), hub)
