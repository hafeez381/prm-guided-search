exec(open(__import__("glob").glob("/kaggle/input/**/bootstrap.py", recursive=True)[0]).read())

import json
import platform
import time

import accelerate
import bitsandbytes
import torch
import transformers

from src.generation import format_step_prompt
from src.extraction import extract_answer, parse_steps
from src.utils import GREEDY_GEN_KWARGS, load_models, make_hf_generate_fn, set_seed_for_problem

print("python", platform.python_version(), "torch", torch.__version__, "transformers", transformers.__version__,
      "accelerate", accelerate.__version__, "bitsandbytes", bitsandbytes.__version__)
print("device_count", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

t0 = time.perf_counter()
gen_model, gen_tok, judge_model, judge_tok = load_models("cuda:0", "cuda:1")
print(f"load_seconds {time.perf_counter() - t0:.1f}")
print("generator_device", gen_model.device, "judge_device", judge_model.device)
for i in range(torch.cuda.device_count()):
    print(f"cuda:{i} allocated_GB {torch.cuda.memory_allocated(i) / 1e9:.2f}")

for word in [" VALID", " INVALID", "VALID", "INVALID"]:
    print("judge_tokens", repr(word), judge_tok.encode(word, add_special_tokens=False))

problem = json.loads(open("data/gsm8k_test_50.jsonl").readline())
set_seed_for_problem(0)
gen_fn = make_hf_generate_fn(gen_model, gen_tok)
t0 = time.perf_counter()
out = gen_fn(format_step_prompt(problem["question"]), 1, dict(GREEDY_GEN_KWARGS))[0]
print(f"greedy_seconds {time.perf_counter() - t0:.1f} num_tokens {out['num_tokens']}")
print("----- completion -----\n" + out["text"] + "\n----------------------")
print("steps", len(parse_steps(out["text"])), "answer", extract_answer(out["text"]))
