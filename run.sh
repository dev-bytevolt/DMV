#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

usage() {
  echo "Usage: $(basename "$0") [--skip-processed] <pdf-file> [pdf-file ...]" >&2
  echo "       $(basename "$0") [--skip-processed] <directory-with-pdfs>" >&2
  echo "       $(basename "$0") [--skip-processed] ./artifacts/input/*" >&2
  echo >&2
  echo "  --skip-processed  Skip PDFs that already have artifacts/<name>/output.pdf" >&2
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

pdf_files=()
extra_flags=()

resolve_path() {
  local path="$1"
  if [[ "$path" != /* ]]; then
    path="$(cd "$(dirname "$path")" && pwd)/$(basename "$path")"
  fi
  printf '%s' "$path"
}

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    --skip-processed|-v|--verbose)
      extra_flags+=("$arg")
      continue
      ;;
    --env-file)
      echo "Error: pass --env-file via dmv directly; run.sh always uses $ROOT/.env" >&2
      exit 1
      ;;
    -*)
      echo "Error: unknown flag: $arg" >&2
      usage
      exit 1
      ;;
  esac

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

# Bash 3.2 + set -u treats an empty "${array[@]}" as unbound.
exec dmv \
  --env-file "$ROOT/.env" \
  -v \
  ${extra_flags[@]+"${extra_flags[@]}"} \
  "${pdf_files[@]}"
