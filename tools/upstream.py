"""tools/upstream.py: what a Renovate PR actually changed — the release facts between old and new.

Stdlib only. Diffs the lockfiles of two trees (Cargo.lock, and the uv.lock of each Python tool) and writes into the
output directory: `versions.txt` (lock: package old -> new, per line), `release-notes.md` (the GitHub release of
every moved direct dependency, when reachable), and the public API of the engine and the library the tools wrap
(ort, tokenizers: `cargo public-api` over the published crates; gliner: the wheel's public functions, classes and
methods): `api-added.txt`, `api-removed.txt` (one `package: item` per line) and `api-diff/<package>.diff`.
`api-added.txt` is what maintenance reads to decide whether there is an `expose` to do. Exit 0 always: context is
never a failure; what could not be fetched is a line saying so."""

import argparse
import ast
import difflib
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from enum import StrEnum, auto
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from types import MappingProxyType
from typing import Final, NamedTuple, NoReturn

NAME_RE: Final = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")
GITHUB_RE: Final = re.compile(r"github\.com[/:]([^/]+/[^/#?.]+)")
TIMEOUT: Final = 60
USER_AGENT: Final = "gliner-upstream (https://github.com/edamlabs/gliner)"
UNREACHABLE: Final = (
    OSError,
    ValueError,
    KeyError,
    subprocess.CalledProcessError,
    tarfile.TarError,
    zipfile.BadZipFile,
)


class Registry(StrEnum):
    CRATES = auto()
    PYPI = auto()


class Move(NamedTuple):
    lock: str
    registry: Registry
    name: str
    old: str
    new: str


LOCKS: Final = MappingProxyType(
    {
        "Cargo.lock": Registry.CRATES,
        "tools/export/uv.lock": Registry.PYPI,
        "tools/train/uv.lock": Registry.PYPI,
        "tools/uv.lock": Registry.PYPI,
    }
)
API: Final = MappingProxyType({"ort": Registry.CRATES, "tokenizers": Registry.CRATES, "gliner": Registry.PYPI})

type Fetch = Callable[[str], bytes]
type Surface = Callable[[str, str], list[str]]


def fetch(url: str, token: str = "") -> bytes:
    """A GET with the house identity; the token only ever travels to the GitHub API."""
    auth = {"Authorization": f"Bearer {token}"} if token and url.startswith("https://api.github.com/") else {}
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **auth})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read()


def unreachable(name: str) -> NoReturn:
    raise LookupError(f"no public API source for {name}")


def pins(path: Path) -> dict[str, str]:
    """package -> its locked versions (comma-joined when a lock holds several), for every package in a lockfile."""
    packages = tomllib.loads(path.read_text()).get("package", []) if path.is_file() else []
    by_name = groupby(sorted(packages, key=itemgetter("name")), key=itemgetter("name"))
    return {name: ", ".join(sorted(p["version"] for p in group if "version" in p)) for name, group in by_name}


def moved(base: Path, head: Path) -> list[Move]:
    """Every package locked in both trees whose versions differ, lock by lock."""
    return [
        Move(lock, registry, name, old, new)
        for lock, registry in LOCKS.items()
        for before, after in [(pins(base / lock), pins(head / lock))]
        for name, old in before.items()
        if (new := after.get(name)) is not None and new != old
    ]


def direct(root: Path) -> set[str]:
    """The names the project itself declares: Cargo.toml dependencies, and every tool's pyproject dependencies."""
    cargo = tomllib.loads((root / "Cargo.toml").read_text())
    crates = {name for table in ("dependencies", "dev-dependencies") for name in cargo.get(table, {})}
    projects = [tomllib.loads(path.read_text()) for path in sorted(root.glob("tools/*/pyproject.toml"))]
    specs = [
        spec
        for data in projects
        for group in [data.get("project", {}).get("dependencies", ()), *data.get("dependency-groups", {}).values()]
        for spec in group
    ]
    return crates | {match.group(1).lower() for spec in specs if (match := NAME_RE.match(str(spec)))}


def python_surface(sources: Mapping[str, str]) -> list[str]:
    """Public functions, classes and their public methods (with signatures) of a package's source files."""
    modules = {
        path.removesuffix(".py").removesuffix("/__init__").replace("/", "."): ast.parse(text)
        for path, text in sources.items()
        if path.endswith(".py") and not any(part.startswith("_") and part != "__init__.py" for part in path.split("/"))
    }
    return sorted(item for module, tree in modules.items() for node in tree.body for item in _surface(module, node))


def _surface(prefix: str, node: ast.stmt) -> list[str]:
    """One top-level node as API lines: a public function with its signature, a public class and its methods."""
    match node:
        case ast.FunctionDef(name=name, args=args) | ast.AsyncFunctionDef(name=name, args=args) if public(name):
            return [f"{prefix}.{name}({ast.unparse(args)})"]
        case ast.ClassDef(name=name, body=body) if not name.startswith("_"):
            return [f"{prefix}.{name}", *(line for child in body for line in _surface(f"{prefix}.{name}", child))]
        case _:
            return []


def public(name: str) -> bool:
    """A public name, or the constructor (its signature is API too)."""
    return not name.startswith("_") or name == "__init__"


def wheel_surface(name: str, version: str, get: Fetch) -> list[str]:
    """The public surface of a PyPI release, read from its wheel."""
    release = json.loads(get(f"https://pypi.org/pypi/{name}/{version}/json"))
    url = next(item["url"] for item in release["urls"] if item["packagetype"] == "bdist_wheel")
    with zipfile.ZipFile(io.BytesIO(get(url))) as wheel:
        return python_surface({path: wheel.read(path).decode() for path in wheel.namelist() if path.endswith(".py")})


def crate_surface(name: str, version: str, get: Fetch) -> list[str]:
    """The public API of a published crate, as `cargo public-api` prints it (nightly rustdoc JSON)."""
    with tempfile.TemporaryDirectory() as work:
        with tarfile.open(
            fileobj=io.BytesIO(get(f"https://static.crates.io/crates/{name}/{name}-{version}.crate"))
        ) as crate:
            crate.extractall(work, filter="data")
        manifest = Path(work) / f"{name}-{version}" / "Cargo.toml"
        command = ["cargo", "+nightly", "public-api", "--manifest-path", str(manifest), "-sss"]
        return sorted(subprocess.run(command, check=True, capture_output=True, text=True).stdout.splitlines())


SURFACES: Final = MappingProxyType({Registry.CRATES: crate_surface, Registry.PYPI: wheel_surface})


def repository(move: Move, get: Fetch) -> str:
    """The GitHub owner/repo a package declares on its registry, or empty."""
    match move.registry:
        case Registry.CRATES:
            links = [json.loads(get(f"https://crates.io/api/v1/crates/{move.name}"))["crate"].get("repository")]
        case Registry.PYPI:
            info = json.loads(get(f"https://pypi.org/pypi/{move.name}/json"))["info"]
            links = [*(info.get("project_urls") or {}).values(), info.get("home_page")]
    return next((found.group(1) for link in links if (found := GITHUB_RE.search(str(link or "")))), "")


def notes_for(move: Move, token: str) -> str:
    """The GitHub release whose tag names the new version, from the package's declared repository."""
    get = lambda url: fetch(url, token)  # noqa: E731 — the token travels with every call of this one note
    version = move.new.split(", ")[-1]
    try:
        repo = repository(move, get)
        tags = json.loads(get(f"https://api.github.com/repos/{repo}/tags?per_page=100"))
        tag = next((t["name"] for t in tags if t["name"].removeprefix("v").endswith(version)), f"v{version}")
        body = json.loads(get(f"https://api.github.com/repos/{repo}/releases/tags/{tag}")).get("body") or ""
    except UNREACHABLE as error:
        return f"## {move.name} {move.new}\n\n_No release notes ({error})._\n"
    return f"## {move.name} {move.new} ({repo} {tag})\n\n{body.strip()}\n"


def _run_api(move: Move, surface: Surface, output: Path) -> tuple[list[str], list[str]]:
    """Added and removed items of one moved API package; its unified diff, or why there is none, on disk."""
    target = output / "api-diff" / f"{move.name}.diff"
    try:
        before, after = surface(move.name, move.old), surface(move.name, move.new)
    except (*UNREACHABLE, LookupError) as error:
        target.write_text(f"API diff unavailable: {error}\n")
        return [], []
    diff = difflib.unified_diff(before, after, f"{move.name} {move.old}", f"{move.name} {move.new}", lineterm="")
    target.write_text("\n".join(diff) + "\n")
    old, new = set(before), set(after)
    return [f"{move.name}: {item}" for item in after if item not in old], [
        f"{move.name}: {item}" for item in before if item not in new
    ]


def run(base: Path, head: Path, output: Path, surface: Surface, notes: Callable[[Move], str]) -> int:
    """Write every context file; context is never a failure."""
    moves = moved(base, head)
    declared = direct(head)
    (output / "api-diff").mkdir(parents=True, exist_ok=True)
    apis = [_run_api(move, surface, output) for move in moves if move.name in API and API[move.name] is move.registry]
    (output / "versions.txt").write_text("".join(f"{m.lock}: {m.name} {m.old} -> {m.new}\n" for m in moves))
    (output / "release-notes.md").write_text("\n".join(notes(m) for m in moves if m.name in declared))
    (output / "api-added.txt").write_text("".join(f"{line}\n" for added, _ in apis for line in added))
    (output / "api-removed.txt").write_text("".join(f"{line}\n" for _, removed in apis for line in removed))
    print(f"{len(moves)} packages moved, {sum(len(added) for added, _ in apis)} API items added")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="a tree holding the base branch's lockfiles")
    parser.add_argument("--head", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token", default="", help="GitHub token — raises the API rate limit")
    args = parser.parse_args()
    get = lambda url: fetch(url, args.token)  # noqa: E731
    return run(
        args.base,
        args.head,
        args.output,
        lambda name, version: SURFACES[API[name]](name, version, get),
        lambda move: notes_for(move, args.token),
    )


if __name__ == "__main__":
    sys.exit(main())
