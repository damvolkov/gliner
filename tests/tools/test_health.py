from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tools import health
from tools.upstream import Registry


def test_requirements_cargo_and_tools(locks: Path) -> None:
    assert health.requirements(locks / "head") == [
        (Registry.CRATES, "ort"),
        (Registry.CRATES, "tokenizers"),
        (Registry.PYPI, "gliner"),
        (Registry.PYPI, "ruff"),
        (Registry.PYPI, "torch"),
    ]


@pytest.mark.parametrize(
    ("registry_name", "name", "verdicts"),
    [
        (Registry.CRATES, "alive", []),
        (Registry.CRATES, "gone", ["vanished"]),
        (Registry.CRATES, "old", ["stale"]),
        (Registry.CRATES, "boxed", ["archived"]),
        (Registry.PYPI, "fresh", []),
        (Registry.PYPI, "gone", ["vanished"]),
        (Registry.PYPI, "sleepy", ["dormant"]),
    ],
)
def test_check_verdict(
    registry: Callable[[str], Any], now: float, registry_name: Registry, name: str, verdicts: list[str]
) -> None:
    assert [row[2] for row in health.check(registry_name, name, registry, now)] == verdicts


@pytest.mark.parametrize(
    ("deps", "expected"), [('alive = "1"\n', 0), ('old = "1"\n', health.FINDINGS_EXIT)], ids=["healthy", "findings"]
)
def test_run_exit(registry: Callable[[str], Any], now: float, tmp_path: Path, deps: str, expected: int) -> None:
    (tmp_path / "Cargo.toml").write_text(f'[package]\nname = "x"\n\n[dependencies]\n{deps}')
    assert health.run(tmp_path, tmp_path / "report.md", registry, now) == expected
