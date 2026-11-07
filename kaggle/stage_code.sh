#!/usr/bin/env bash
# Build the Kaggle code dataset from the committed tree only: code.tar (git archive HEAD + GIT_SHA) and bootstrap.py.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git diff --quiet HEAD -- src submission_adapter.py data kaggle run_all.py compliance_check.py build_artifacts.py prompts.lock 2>/dev/null || { echo "uncommitted changes in shipped paths"; exit 1; }
out=kaggle/output/code_dataset
rm -rf "$out" && mkdir -p "$out"
git archive --format=tar -o "$out/code.tar" HEAD
git rev-parse HEAD > "$out/GIT_SHA"
tar -rf "$out/code.tar" -C "$out" GIT_SHA && rm "$out/GIT_SHA"
cp kaggle/bootstrap.py "$out/"
cat > "$out/dataset-metadata.json" <<JSON
{"title": "prm-guided-search-code", "id": "abdul381/prm-guided-search-code", "licenses": [{"name": "CC0-1.0"}]}
JSON
shasum -a 256 "$out/code.tar" "$out/bootstrap.py"
