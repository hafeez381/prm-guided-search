#!/usr/bin/env bash
# Usage: kaggle/push_job.sh <job-name> <command...>
# Ships HEAD as a new version of the private code dataset, then runs <command> from the repo root on Kaggle T4 x2.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
job=$1; shift
./kaggle/stage_code.sh
kaggle datasets version -p kaggle/output/code_dataset -m "$(git rev-parse --short HEAD)" -q >/dev/null
until kaggle datasets status abdul381/prm-guided-search-code 2>/dev/null | grep -q ready; do sleep 10; done
dir=kaggle/output/jobs/$job && rm -rf "$dir" && mkdir -p "$dir"
cat > "$dir/kernel.py" <<PY
import glob, subprocess
exec(open(glob.glob("/kaggle/input/**/bootstrap.py", recursive=True)[0]).read())
expected = "$(git rev-parse HEAD)"
assert open("GIT_SHA").read().strip() == expected, "stale code dataset"
subprocess.run($(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@"), check=True)
PY
cat > "$dir/kernel-metadata.json" <<JSON
{"id": "abdul381/prm-guided-search-$job", "title": "prm-guided-search $job", "code_file": "kernel.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": ["abdul381/prm-guided-search-code"], "competition_sources": [], "kernel_sources": []}
JSON
kaggle kernels push -p "$dir" --accelerator NvidiaTeslaT4 --timeout 10800 2>&1 | grep -v outdated
