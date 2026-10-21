import io
import json
import zipfile
from pathlib import Path

import pytest

from tools import upstream
from tools.upstream import Move, Registry

SOURCES = {
    "pkg/__init__.py": "from pkg.model import GLiNER\n\ndef top(a, b=1): ...\n\n_hidden = 1\n",
    "pkg/model.py": "class GLiNER:\n    def predict(self, text, *, threshold=0.5): ...\n    def _inner(self): ...\n",
    "pkg/_private.py": "def secret(): ...\n",
    "pkg/data/README.txt": "not python\n",
}


def test_pins_merge_duplicate_names(locks: Path) -> None:
    assert upstream.pins(locks / "head" / "Cargo.lock") == {
        "gliner": "0.1.0",
        "ort": "2.0.0-rc.16",
        "tokenizers": "0.23.2",
        "syn": "1.0.109, 2.0.101",
    }


def test_pins_missing_lock(tmp_path: Path) -> None:
    assert upstream.pins(tmp_path / "Cargo.lock") == {}


def test_moved_every_lock(locks: Path) -> None:
    assert upstream.moved(locks / "base", locks / "head") == [
        Move("Cargo.lock", Registry.CRATES, "ort", "2.0.0-rc.15", "2.0.0-rc.16"),
        Move("Cargo.lock", Registry.CRATES, "syn", "1.0.109, 2.0.100", "1.0.109, 2.0.101"),
        Move("tools/export/uv.lock", Registry.PYPI, "gliner", "0.2.29", "0.2.30"),
    ]


def test_direct_dependencies(locks: Path) -> None:
    assert upstream.direct(locks / "head") == {"ort", "tokenizers", "tower", "gliner", "torch", "ruff"}


def test_python_surface_public_only() -> None:
    assert upstream.python_surface(SOURCES) == [
        "pkg.model.GLiNER",
        "pkg.model.GLiNER.predict(self, text, *, threshold=0.5)",
        "pkg.top(a, b=1)",
    ]


def test_wheel_surface_from_pypi() -> None:
    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as archive:
        [archive.writestr(name, text) for name, text in SOURCES.items()]
    files = {
        "https://pypi.org/pypi/pkg/1.0/json": json.dumps(
            {"urls": [{"packagetype": "sdist", "url": "s"}, {"packagetype": "bdist_wheel", "url": "w"}]}
        ).encode(),
        "w": wheel.getvalue(),
    }
    assert upstream.wheel_surface("pkg", "1.0", files.__getitem__)[-1] == "pkg.top(a, b=1)"


def test_run_writes_context(locks: Path, tmp_path: Path) -> None:
    surfaces = {
        ("ort", "2.0.0-rc.15"): ["pub fn ort::a()", "pub fn ort::gone()"],
        ("ort", "2.0.0-rc.16"): ["pub fn ort::a()", "pub fn ort::b()"],
    }

    def surface(name: str, version: str) -> list[str]:
        return surfaces[name, version] if (name, version) in surfaces else upstream.unreachable(name)

    out = tmp_path / "context"
    assert upstream.run(locks / "base", locks / "head", out, surface, lambda move: f"## {move.name}\n") == 0
    assert (out / "versions.txt").read_text().splitlines()[0] == "Cargo.lock: ort 2.0.0-rc.15 -> 2.0.0-rc.16"
    assert (out / "api-added.txt").read_text() == "ort: pub fn ort::b()\n"
    assert (out / "api-removed.txt").read_text() == "ort: pub fn ort::gone()\n"
    assert (out / "api-diff" / "gliner.diff").read_text().startswith("API diff unavailable:")
    assert (out / "release-notes.md").read_text() == "## ort\n\n## gliner\n"


@pytest.mark.parametrize(("name", "registry"), [("ort", Registry.CRATES), ("gliner", Registry.PYPI)])
def test_api_registry(name: str, registry: Registry) -> None:
    assert upstream.API[name] is registry
