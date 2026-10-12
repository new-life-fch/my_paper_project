#!/usr/bin/env bash
# =============================================================================
# Migrate results/cache/ from the pre-2026-10-12 names to the canonical scheme.
#
#   canonical: results/cache/{tag}_q{N}_{instruct,judge}   (docs/CACHE_NAMING.md)
#   legacy:    results/cache/q500_instruct, llama31_8b_instruct, ...
#
# The alias table lives in ONE place -- src/cache_paths.py -- and this script
# reads it from there, so the two can never drift apart.
#
# Usage:
#   bash scripts/migrate_cache_names.sh            # dry run (default): list only
#   bash scripts/migrate_cache_names.sh --apply    # actually rename
#
# ⚠️  results/cache may be a SYMLINK to the real cache on the data disk
#     (CLAUDE.md §七 item 5). This script only ever `mv`s individual leaf
#     directories inside it and never recurses through the link, but read the
#     dry run before applying.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1
CACHE_DIR="${CACHE_DIR:-results/cache}"

if [ ! -d "$CACHE_DIR" ]; then
  echo "No $CACHE_DIR here -- nothing to migrate (this is expected on a fresh clone:"
  echo "the activation caches are gitignored and live on the GPU host)."
  exit 0
fi
if [ -L "$CACHE_DIR" ]; then
  echo "note: $CACHE_DIR is a symlink -> $(readlink "$CACHE_DIR")"
  echo "      (renaming inside it affects the real cache; that is intended)"
fi

# alias -> canonical, straight from the single source of truth
# pick an interpreter that exists (the GPU host has `python`; a bare macOS
# checkout may only have the project venv or python3)
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; fi
if [ -x ../.venv/bin/python ]; then PY=../.venv/bin/python; fi

PAIRS="$("$PY" - <<'PY'
import os, sys
sys.path.insert(0, os.path.abspath("."))
from src.cache_paths import LEGACY_ALIASES
for canon, alts in sorted(LEGACY_ALIASES.items()):
    for a in alts:
        print(f"{a}\t{canon}")
PY
)"

if [ -z "$PAIRS" ]; then
  echo "could not read the alias table from src/cache_paths.py" >&2
  exit 1
fi

n_hit=0; n_skip=0; n_conflict=0
while IFS=$'\t' read -r old new; do
  [ -z "${old:-}" ] && continue
  op="$CACHE_DIR/$old"; np="$CACHE_DIR/$new"
  if [ -e "$np" ]; then
    printf "  SKIP      %-34s (canonical already exists)\n" "$new"
    n_skip=$((n_skip+1)); continue
  fi
  if [ ! -e "$op" ]; then continue; fi
  if [ -e "$np" ]; then n_conflict=$((n_conflict+1)); continue; fi
  n_hit=$((n_hit+1))
  if [ "$APPLY" = "1" ]; then
    mv "$op" "$np" && printf "  RENAMED   %-34s -> %s\n" "$old" "$new"
  else
    printf "  would mv  %-34s -> %s\n" "$old" "$new"
  fi
done <<< "$PAIRS"

echo
echo "found $n_hit legacy dir(s) to migrate, $n_skip already canonical"
if [ "$APPLY" != "1" ] && [ "$n_hit" -gt 0 ]; then
  echo "dry run -- re-run with --apply to perform the renames"
fi
echo "tip: don't rely on the names at all -- meta.json's 'model' field is authoritative:"
echo "     python -c \"import json,glob;[print(p,json.load(open(p))['model']) for p in sorted(glob.glob('$CACHE_DIR/*/meta.json'))]\""
