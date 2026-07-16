#!/usr/bin/env bash
# Verify the G1 probe under the tactus Lean backend.
# Usage: ./check.sh [--verify-module kir]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERUS="$HERE/../../tactus/source/target-verus/release/verus"

if [[ ! -x "$VERUS" ]]; then
  echo "error: tactus verus binary not found at $VERUS" >&2
  exit 1
fi

exec "$VERUS" --lean-backend --crate-type=lib "$HERE/src/lib.rs" "$@"
