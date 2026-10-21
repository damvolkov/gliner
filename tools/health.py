"""tools/health.py: dependency health — archived, vanished, stale or dormant upstreams, as a Markdown report.

Stdlib only, like every house tool that must run anywhere. Reads the direct dependencies — Cargo.toml's crates and
every tool's pyproject.toml (project dependencies and dependency-groups) — probes crates.io or PyPI for each and,
through the repository the package declares, GitHub. Exit 0: everything alive. Exit 3: findings — the contract the
workflow reads. Any other failure is the probe's own error."""

import argparse
import concurrent.futures
import json
import re
import sys
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from tools.upstream import GITHUB_RE, NAME_RE, Registry

FINDINGS_EXIT: Final = 3
STALE_DAYS: Final = 540
DORMANT_DAYS: Final = 365
CONCURRENCY: Final = 8
TIMEOUT: Final = 30
CRATES_PAUSE: Final = 1.0
USER_AGENT: Final = "gliner-health (https://github.com/damvolkov/gliner)"
DIGITS_RE: Final = re.compile(r"\d+")

type Get = Callable[[str], Any]
type Row = tuple[str, str, str, str]


def version_key(version: str) -> tuple[tuple[int, ...], int]:
    """A release string as a sortable key: numbers compare numerically, a suffix (pre-release) loses ties."""
    return (tuple(int(part) for part in DIGITS_RE.findall(version)), -len(re.sub(r"[\d.]", "", version)))


def days(stamp: str, now: float) -> int:
    """Whole days from an ISO-8601 stamp (naive means UTC) to now."""
    moment = datetime.fromisoformat(stamp)
    return int((now - (moment if moment.tzinfo else moment.replace(tzinfo=UTC)).timestamp()) // 86400)


def get_json(url: str, token: str = "") -> Any:
    """A JSON GET with the house identity; a 404 answers None — vanished is a finding, not a crash."""
    headers = {"User-Agent": USER_AGENT}
    headers |= {"Authorization": f"Bearer {token}"} if token and url.startswith("https://api.github.com/") else {}
    ### crates.io's crawler policy: at most one request per second.
    time.sleep(CRATES_PAUSE if url.startswith("https://crates.io/") else 0)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=TIMEOUT) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        return None


def requirements(root: Path) -> list[tuple[Registry, str]]:
    """Every direct dependency, deduplicated and sorted: Cargo.toml's crates, the tools' Python packages."""
    crates = tomllib.loads((root / "Cargo.toml").read_text()).get("dependencies", {})
    projects = [tomllib.loads(path.read_text()) for path in root.glob("tools/*/pyproject.toml")]
    specs = (
        str(spec)
        for data in projects
        for group in [data.get("project", {}).get("dependencies", ()), *data.get("dependency-groups", {}).values()]
        for spec in group
    )
    python = {match.group(1).lower() for spec in specs if (match := NAME_RE.match(spec))}
    return sorted({*((Registry.CRATES, name) for name in crates), *((Registry.PYPI, name) for name in python)})


def _check_repo(name: str, repo: str, get: Get, now: float) -> list[Row]:
    meta = (get(f"https://api.github.com/repos/{repo}") if repo else None) or {}
    pushed = meta.get("pushed_at") or ""
    idle = days(pushed, now) if pushed else 0
    return [
        *([(name, f"github:{repo}", "archived", "the repository is archived")] if meta.get("archived") else []),
        *([(name, f"github:{repo}", "dormant", f"no push in {idle} days")] if idle > DORMANT_DAYS else []),
    ]


def _check_crate(name: str, get: Get, now: float) -> list[Row]:
    meta = get(f"https://crates.io/api/v1/crates/{name}")
    if meta is None:
        return [(name, "crates.io", "vanished", "crate is gone from the index")]
    released = max((v["created_at"] for v in meta["versions"] if not v["yanked"]), default="")
    age = days(released, now) if released else STALE_DAYS + 1
    stale = [(name, "crates.io", "stale", f"last release is {age} days old")] if age > STALE_DAYS else []
    found = GITHUB_RE.search(meta["crate"].get("repository") or "")
    return stale + _check_repo(name, found.group(1) if found else "", get, now)


def _check_pypi(name: str, get: Get, now: float) -> list[Row]:
    meta = get(f"https://pypi.org/pypi/{name}/json")
    if meta is None:
        return [(name, "PyPI", "vanished", "package is gone from the index")]
    latest = max(meta["releases"], key=version_key, default="")
    uploaded = (meta["releases"].get(latest) or [{}])[0].get("upload_time", "")
    age = days(uploaded, now) if uploaded else STALE_DAYS + 1
    stale = [(name, "PyPI", "stale", f"last release {latest} is {age} days old")] if age > STALE_DAYS else []
    links = (meta.get("info", {}).get("project_urls") or {}).values()
    repo = next((found.group(1) for link in links if (found := GITHUB_RE.search(str(link or "")))), "")
    return stale + _check_repo(name, repo, get, now)


CHECKS: Final = MappingProxyType({Registry.CRATES: _check_crate, Registry.PYPI: _check_pypi})


def check(registry: Registry, name: str, get: Get, now: float) -> list[Row]:
    """One package's findings as (dependency, registry, verdict, detail) rows — empty means alive."""
    return CHECKS[registry](name, get, now)


def run(root: Path, output: Path, get: Get, now: float) -> int:
    """Write the report; crates are probed one at a time (crates.io's policy), PyPI concurrently meanwhile."""
    deps = requirements(root)
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        python = [pool.submit(check, registry, name, get, now) for registry, name in deps if registry is Registry.PYPI]
        crates = [check(registry, name, get, now) for registry, name in deps if registry is Registry.CRATES]
        found = [row for rows in [*crates, *(future.result() for future in python)] for row in rows]
    lines = ["| Dependency | Registry | Verdict | Detail |", "|---|---|---|---|"]
    lines += [f"| {name} | {registry} | {verdict} | {detail} |" for name, registry, verdict, detail in found]
    output.write_text("\n".join(lines) + "\n" if found else f"All {len(deps)} direct dependencies are healthy.\n")
    print(f"{len(deps)} dependencies checked, {len(found)} findings")
    return FINDINGS_EXIT if found else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token", default="", help="GitHub token — raises the API rate limit")
    args = parser.parse_args()
    return run(args.root, args.output, lambda url: get_json(url, args.token), time.time())


if __name__ == "__main__":
    sys.exit(main())
