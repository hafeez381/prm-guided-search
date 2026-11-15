#!/usr/bin/env bash
# Benchmark runs on a Colab GPU through the `colab` CLI, from the repo root. Every CLI call goes through
# colab/colab_retry.py (wall-clock timeout + retries); long work runs detached on the VM.
#
#   bash colab/colab_run.sh new [GPU]            # usage before/after, allocate (L4 default), save the browser URL
#   bash colab/colab_run.sh setup                # ship git-archive HEAD + GIT_SHA, upgrade bitsandbytes
#   bash colab/colab_run.sh launch NAME CMD...   # run CMD detached in the repo; log in /content/logs/NAME.log
#   bash colab/colab_run.sh status NAME          # last log lines, GPU memory/utilization, process alive?
#   bash colab/colab_run.sh sh 'COMMAND'         # one short shell command on the VM
#   bash colab/colab_run.sh fetch REMOTE LOCAL   # download a small file, verify sha256 against the VM's
#   bash colab/colab_run.sh stop                 # release the VM, then show sessions and usage
set -euo pipefail

SESSION="${SESSION:-pa3-gpu}"
REMOTE=/content/repo
RETRY="python3 colab/colab_retry.py"

remote_py() {
  local f; f="$(mktemp -d)/remote.py"; cat > "$f"
  $RETRY "$1" 3 colab exec -s "$SESSION" --timeout "$1" -f "$f"
}

remote_sh() {
  local b64; b64="$(printf '%s' "$2" | base64 | tr -d '\n')"
  remote_py "$1" <<PY
import base64, subprocess
r = subprocess.run(base64.b64decode("$b64").decode(), shell=True, capture_output=True, text=True)
print(r.stdout[-6000:], r.stderr[-3000:]); print("exit", r.returncode)
PY
}

cmd="${1:-}"; shift || true
case "$cmd" in
new)
  colab usage </dev/null
  $RETRY 300 1 colab new -s "$SESSION" --gpu "${1:-L4}"
  mkdir -p runs && colab url -s "$SESSION" </dev/null | tee runs/colab_url.txt
  colab usage </dev/null
  ;;
setup)
  [[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "commit first: the VM runs HEAD" >&2; exit 1; }
  mkdir -p runs && git rev-parse HEAD > runs/GIT_SHA
  git archive --format=tar HEAD > runs/ship.tar && tar -rf runs/ship.tar -C runs GIT_SHA && gzip -f runs/ship.tar
  echo "HEAD $(cat runs/GIT_SHA)  sha256 $(shasum -a 256 runs/ship.tar.gz | cut -d' ' -f1)"
  $RETRY 180 3 colab upload -s "$SESSION" runs/ship.tar.gz /content/ship.tar.gz
  remote_sh 200 "rm -rf $REMOTE && mkdir -p $REMOTE /content/logs /content/out && tar -xzf /content/ship.tar.gz -C $REMOTE && cat $REMOTE/GIT_SHA && pip install -q -U bitsandbytes 2>&1 | tail -2 && python -c 'import torch, transformers, bitsandbytes; print(torch.__version__, transformers.__version__, bitsandbytes.__version__, torch.cuda.get_device_name(0))'"
  ;;
launch)
  name="${1:?NAME required}"; shift
  b64="$(printf '%s' "$*" | base64 | tr -d '\n')"
  remote_py 60 <<PY
import base64, shlex, subprocess, time
cmd = base64.b64decode("$b64").decode()
print("launching:", cmd)
subprocess.Popen(f"cd $REMOTE && TOKENIZERS_PARALLELISM=false nohup bash -c {shlex.quote(cmd)} > /content/logs/$name.log 2>&1; echo EXIT=\$? >> /content/logs/$name.log",
                 shell=True, start_new_session=True)
time.sleep(15)
print(subprocess.run("pgrep -af run_all || echo NOT RUNNING", shell=True, capture_output=True, text=True).stdout)
PY
  ;;
status)
  name="${1:?NAME required}"
  remote_sh 60 "tail -n 6 /content/logs/$name.log | cut -c1-300; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"
  ;;
sh)
  remote_sh 120 "cd $REMOTE && $1"
  ;;
fetch)
  src="${1:?REMOTE path required}"; dst="${2:?LOCAL path required}"
  mkdir -p "$(dirname "$dst")"
  remote_sh 60 "sha256sum $src"
  $RETRY 230 3 colab download -s "$SESSION" "$src" "$dst"
  echo "local: $(shasum -a 256 "$dst" | cut -d' ' -f1)  (must equal the remote line above)"
  ;;
stop)
  $RETRY 120 3 colab stop -s "$SESSION" || true
  colab sessions </dev/null
  colab usage </dev/null
  ;;
*)
  sed -n '2,11p' "$0"; exit 1
  ;;
esac
