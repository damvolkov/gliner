#!/usr/bin/env bash
# Dependency health of the direct crates: vanished from crates.io, last release older than STALE_DAYS, repository
# archived, or no push in DORMANT_DAYS. Markdown report on stdout; exit 0 when healthy, 3 with findings.
# Needs cargo, curl, jq; GH_TOKEN (optional) lifts the GitHub API rate limit.
set -euo pipefail

STALE_DAYS="${STALE_DAYS:-540}"
DORMANT_DAYS="${DORMANT_DAYS:-365}"
AGENT="gliner-health (https://github.com/damvolkov/gliner)"
NOW="$(date +%s)"
findings=()

days() { echo $(((NOW - $(date -d "$1" +%s)) / 86400)); }

github() {
  local auth=()
  [ -z "${GH_TOKEN:-}" ] || auth=(-H "Authorization: Bearer ${GH_TOKEN}")
  curl -fsSL -H "Accept: application/vnd.github+json" "${auth[@]}" "https://api.github.com/repos/$1"
}

crates="$(cargo metadata --format-version 1 --no-deps --locked |
  jq -r '.packages[0].dependencies[] | select(.kind == null) | .name' | sort -u)"

for crate in $crates; do
  sleep 1
  if ! meta="$(curl -fsSL -A "$AGENT" "https://crates.io/api/v1/crates/${crate}")"; then
    findings+=("| ${crate} | missing | not found on crates.io |")
    continue
  fi
  released="$(jq -r '[.versions[] | select(.yanked | not)][0].created_at' <<<"$meta")"
  age="$(days "$released")"
  ((age <= STALE_DAYS)) || findings+=("| ${crate} | stale | last release ${age} days ago (${released%%T*}) |")
  repository="$(jq -r '.crate.repository // ""' <<<"$meta" |
    sed -nE 's#.*github\.com[/:]([^/]+/[^/#?]+).*#\1#p' | sed 's/\.git$//')"
  [ -n "$repository" ] || continue
  info="$(github "$repository")" || continue
  [ "$(jq -r .archived <<<"$info")" != true ] || findings+=("| ${crate} | archived | \`${repository}\` is archived |")
  pushed="$(jq -r .pushed_at <<<"$info")"
  idle="$(days "$pushed")"
  ((idle <= DORMANT_DAYS)) || findings+=("| ${crate} | dormant | last push ${idle} days ago (${pushed%%T*}) |")
done

if [ "${#findings[@]}" -eq 0 ]; then
  echo "All $(wc -w <<<"$crates") direct dependencies are healthy."
  exit 0
fi
printf '| Dependency | Verdict | Detail |\n|---|---|---|\n'
printf '%s\n' "${findings[@]}"
exit 3
