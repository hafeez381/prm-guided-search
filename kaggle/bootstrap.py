"""Shared Kaggle preamble: install bitsandbytes, unpack the code dataset, chdir into it."""
import glob
import os
import subprocess
import sys
import tarfile

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes"], check=True)

WORK = "/kaggle/working/code"
if not os.path.isdir(WORK):
    tars = glob.glob("/kaggle/input/**/code.tar", recursive=True)
    if tars:
        with tarfile.open(tars[0]) as tar:
            tar.extractall(WORK)
    else:
        roots = [os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/src/utils.py", recursive=True)]
        assert roots, "code dataset not mounted"
        subprocess.run(["cp", "-r", roots[0], WORK], check=True)
os.chdir(WORK)
sys.path.insert(0, WORK)
print("GIT_SHA", open("GIT_SHA").read().strip())
