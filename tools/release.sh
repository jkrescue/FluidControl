#!/bin/bash
# release.sh <tag> <project_root>: export a git tag to <project_root>/code_releases/<tag> (read-only) with a SHA-256
# manifest of every .py file. Training/evaluation runs copy and verify exactly this tree.
set -e
TAG=$1; P=$2; D=$P/code_releases/$TAG
[ -e "$D" ] && { echo "exists: $D"; exit 1; }
mkdir -p "$D"; git -C "$(dirname "$0")/.." archive "$TAG" | tar -x -C "$D"
(cd "$D" && python3 -c "
import hashlib, json, pathlib
m = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(pathlib.Path('.').rglob('*.py'))}
pathlib.Path('SOURCE_MANIFEST.json').write_text(json.dumps(m, indent=1)); print(len(m), 'files')")
chmod -R a-w "$D"; echo "released $TAG -> $D"
