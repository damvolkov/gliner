import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

RESOURCES = Path(__file__).parents[1] / "resources"
QUERY = ("author", "search", "filter")


@pytest.fixture
def hub_config() -> Path:
    return RESOURCES / "hub" / "models.toml"


@pytest.fixture
def hub() -> Callable[[str], Any]:
    """The Hub's public API replayed from a fixture: a search keyed by its track, a model or a file by its path."""
    responses = json.loads((RESOURCES / "hub" / "responses.json").read_text())

    def get(url: str) -> Any:
        parts = urlsplit(url)
        query = parse_qs(parts.query)
        track = "&".join(f"{key}={query[key][0]}" for key in QUERY if key in query)
        return responses.get(f"{parts.path}?{track}" if track else parts.path)

    return get


@pytest.fixture
def locks() -> Path:
    """Two trees, base and head: a Cargo.lock and a tools uv.lock each, with a moved engine and a moved library."""
    return RESOURCES / "upstream"


NOW = 1792540800.0  # 2026-10-21T00:00:00Z


@pytest.fixture
def now() -> float:
    return NOW


@pytest.fixture
def registry() -> Callable[[str], Any]:
    """crates.io, PyPI and GitHub replayed: one package per verdict; an unknown URL is a 404 (None)."""
    fresh, old = "2026-09-01T00:00:00Z", "2024-01-01T00:00:00Z"
    responses = {
        "https://crates.io/api/v1/crates/alive": {
            "crate": {"repository": "https://github.com/o/alive"},
            "versions": [
                {"yanked": True, "created_at": fresh},
                {"yanked": False, "created_at": "2026-08-01T00:00:00Z"},
            ],
        },
        "https://crates.io/api/v1/crates/old": {
            "crate": {"repository": None},
            "versions": [{"yanked": False, "created_at": old}],
        },
        "https://crates.io/api/v1/crates/boxed": {
            "crate": {"repository": "https://github.com/o/boxed.git"},
            "versions": [{"yanked": False, "created_at": fresh}],
        },
        "https://pypi.org/pypi/fresh/json": {
            "info": {"project_urls": {"Source": "https://github.com/o/fresh"}},
            "releases": {
                "1.0": [{"upload_time": "2020-01-01T00:00:00"}],
                "1.10": [{"upload_time": "2026-09-01T00:00:00"}],
            },
        },
        "https://pypi.org/pypi/sleepy/json": {
            "info": {"project_urls": {"Source": "https://github.com/o/sleepy"}},
            "releases": {"2.0": [{"upload_time": "2026-09-01T00:00:00"}]},
        },
        "https://api.github.com/repos/o/alive": {"archived": False, "pushed_at": fresh},
        "https://api.github.com/repos/o/boxed": {"archived": True, "pushed_at": fresh},
        "https://api.github.com/repos/o/fresh": {"archived": False, "pushed_at": fresh},
        "https://api.github.com/repos/o/sleepy": {"archived": False, "pushed_at": old},
    }
    return responses.get
