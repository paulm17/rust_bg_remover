#!/bin/sh
set -eu
# A commercial-safe dependency graph must not activate the BRIA adapter.
tree=$(cargo tree -p bgremove-bench --no-default-features -e features --offline --locked)
if printf '%s\n' "$tree" | grep -Eq 'bgremove-ort feature "bria"'; then
  echo "BRIA feature unexpectedly enabled in commercial-safe graph" >&2
  exit 1
fi
printf '%s\n' "$tree" | grep -q 'bgremove-ort'
