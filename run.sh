#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $(basename "$0") <pdf-file> [pdf-file ...]" >&2
  echo "       $(basename "$0") <directory-with-pdfs>" >&2
  echo "       $(basename "$0") ./artifacts/input/*" >&2
  exit 1
fi

pdf_files=()

resolve_path() {
  local path="$1"
  if [[ "$path" != /* ]]; then
    path="$(cd "$(dirname "$path")" && pwd)/$(basename "$path")"
  fi
  printf '%s' "$path"
}

for arg in "$@"; do
  if [[ -d "$arg" ]]; then
    dir="$(resolve_path "$arg")"
    shopt -s nullglob
    for pdf in "$dir"/*.pdf; do
      pdf_files+=("$pdf")
    done
    shopt -u nullglob
    continue
  fi

  if [[ ! -f "$arg" ]]; then
    echo "Error: file not found: $arg" >&2
    exit 1
  fi

  pdf_files+=("$(resolve_path "$arg")")
done

if [[ ${#pdf_files[@]} -eq 0 ]]; then
  echo "Error: no PDF files found" >&2
  exit 1
fi

exec dmv \
  --env-file "$ROOT/.env" \
  -v \
  "${pdf_files[@]}"
