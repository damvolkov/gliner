"""tools/models.py: model watch — GLiNER-family NER models on the Hugging Face Hub newer than the served base.

Stdlib only, like every house tool that must run anywhere. Reads tools/models.toml (the served base, the licences
accepted, the tracked authors and families, the ids a human already triaged), queries the Hub's public API (no auth),
keeps what was created after the base under an accepted licence, and reads each candidate's gliner_config.json to
tell whether this service can serve it: a bi-encoder in span mode, whitespace splitter, first sub-token pooling.
Exit 0: nothing new. Exit 3: candidates — the contract models.yml reads. Any other failure is the probe's own."""

import argparse
import concurrent.futures
import fnmatch
import json
import sys
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlencode

FINDINGS_EXIT: Final = 3
HUB: Final = "https://huggingface.co"
EXPAND: Final = ("createdAt", "lastModified", "downloads", "cardData", "safetensors", "tags")
LIMIT: Final = 1000
CONCURRENCY: Final = 8
TIMEOUT: Final = 30
ABSENT: Final = frozenset({401, 403, 404})
USER_AGENT: Final = "gliner-models (https://github.com/edamlabs/gliner)"
CONTRACT: Final = "bi-encoder, span mode, whitespace words, first sub-token"
HEADER: Final = (
    "| Model | Created | License | Downloads | Params | Architecture | Served here |",
    "|---|---|---|---|---|---|---|",
)

type Get = Callable[[str], Any]


@dataclass(frozen=True, slots=True)
class Watch:
    """What to watch: the served base, the licences accepted, the Hub queries, the ids already triaged."""

    base: str
    licenses: frozenset[str]
    tracks: tuple[Mapping[str, Any], ...]
    ignore: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Candidate:
    """One model worth a human look."""

    id: str
    created: str
    license: str
    downloads: int
    params: int | None
    architecture: str
    supported: bool


def get_json(url: str) -> Any:
    """A JSON GET; a missing, gated or private resource answers None — absence is data, not a crash."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        ### Only absence is swallowed; a 5xx or a rate limit fails the run loudly.
        if error.code not in ABSENT:
            raise
        return None


def load(path: Path) -> Watch:
    """The watch config, read once."""
    data = tomllib.loads(path.read_text())
    return Watch(
        base=data["base"],
        licenses=frozenset(name.lower() for name in data["licenses"]),
        tracks=tuple(data["track"]),
        ignore=tuple(data.get("ignore", ())),
    )


def search_url(track: Mapping[str, Any]) -> str:
    """The Hub search a track describes, newest first, with the fields the report needs."""
    query = [(key, str(value)) for key, value in track.items() if key != "min_downloads"]
    query += [("limit", str(LIMIT)), ("sort", "createdAt"), ("direction", "-1")]
    return f"{HUB}/api/models?{urlencode(query + [('expand[]', field) for field in EXPAND])}"


def created_of(model: Mapping[str, Any]) -> str:
    """When a model appeared; lastModified stands in when the Hub omits createdAt."""
    return str(model.get("createdAt") or model.get("lastModified") or "")


def license_of(model: Mapping[str, Any]) -> str:
    """The licence the card declares, else its `license:` tag."""
    card = (model.get("cardData") or {}).get("license")
    tag = next((tag.removeprefix("license:") for tag in model.get("tags") or () if tag.startswith("license:")), "")
    return str(card or tag or "unknown").lower()


def architecture(config: Mapping[str, Any] | None) -> tuple[str, bool]:
    """The architecture a gliner_config.json declares, and whether this runtime serves it (GLiNER's defaults apply)."""
    match config:
        case None:
            return "unknown: no gliner_config.json", False
        case _:
            encoder = "bi-encoder" if config.get("labels_encoder") else "uni-encoder"
            span = config.get("span_mode", "markerV0")
            mode = "token" if span == "token_level" else f"span ({span})"
            words = (config.get("words_splitter_type", "whitespace"), config.get("subtoken_pooling", "first"))
            contract = words == ("whitespace", "first")
            detail = "" if contract else f" · {words[0]}/{words[1]}"
            return f"{encoder} · {mode}{detail}", encoder == "bi-encoder" and mode != "token" and contract


def _keep(model: Mapping[str, Any], track: Mapping[str, Any], since: str, watch: Watch) -> bool:
    """Newer than the base, an accepted licence, enough downloads for its track, not triaged already."""
    return (
        created_of(model) > since
        and license_of(model) in watch.licenses
        and int(model.get("downloads") or 0) >= int(track.get("min_downloads", 0))
        and model["id"] != watch.base
        and not any(fnmatch.fnmatchcase(model["id"], pattern) for pattern in watch.ignore)
    )


def _candidate(model: Mapping[str, Any], get: Get) -> Candidate:
    name, supported = architecture(get(f"{HUB}/{model['id']}/resolve/main/gliner_config.json"))
    return Candidate(
        id=model["id"],
        created=created_of(model)[:10],
        license=license_of(model),
        downloads=int(model.get("downloads") or 0),
        params=(model.get("safetensors") or {}).get("total"),
        architecture=name,
        supported=supported,
    )


def candidates(watch: Watch, get: Get) -> list[Candidate]:
    """Every tracked model newer than the base, deduplicated across tracks; servable here first, then newest first."""
    ### Without the base's date every model would count as newer: absence is an error here.
    if (base := get(f"{HUB}/api/models/{watch.base}")) is None:
        raise LookupError(f"the served base {watch.base} is not on the Hub")
    since = created_of(base)
    found = {
        model["id"]: model
        for track in watch.tracks
        for model in get(search_url(track)) or ()
        if _keep(model, track, since, watch)
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        listed = list(pool.map(lambda model: _candidate(model, get), found.values()))
    return sorted(listed, key=attrgetter("supported", "created"), reverse=True)


def report(found: list[Candidate]) -> list[str]:
    """A one-line summary, then the Markdown table, one row per candidate."""
    return [
        f"{len(found)} candidates, {sum(c.supported for c in found)} servable here ({CONTRACT}).",
        "",
        *HEADER,
        *(
            f"| [{c.id}]({HUB}/{c.id}) | {c.created} | {c.license} | {c.downloads} "
            f"| {f'{c.params / 1e6:.0f}M' if c.params else '?'} | {c.architecture} | {'yes' if c.supported else 'no'} |"
            for c in found
        ),
    ]


def run(config: Path, output: Path, get: Get) -> int:
    """Write the report; the exit code says whether there is anything new."""
    found = candidates(load(config), get)
    output.write_text("\n".join(report(found)) + "\n" if found else "No GLiNER-family model newer than the base.\n")
    print(f"{len(found)} candidates")
    return FINDINGS_EXIT if found else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("models.toml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.config, args.output, get_json)


if __name__ == "__main__":
    sys.exit(main())
