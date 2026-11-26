"""
Stage 2 - Sequential (continual) fine-tuning of Qwen2.5-0.5B with LoRA.

One LoRA adapter is carried through all tasks:
    stage 1: D1
    stage 2: D2 + buffer(1)
    stage 3: D3 + buffer(1) + buffer(2)
    stage 4: D4 + buffer(1) + buffer(2) + buffer(3)
After every stage the model is evaluated on ALL tasks -> results matrix R[after_task][eval_task].

Replay modes:
    none    no replay (lower bound, shows forgetting)
    random  random subset of each earlier task, same size as the S2L buffer
    s2l     buffers/s2l_<ratio>/<task>.json produced by select_s2l.py
    full    ALL data of earlier tasks (upper bound)

Usage:
    python sequential_finetune.py --debug --replay none      # smoke test first!
    python sequential_finetune.py --replay none
    python sequential_finetune.py --replay random --ratio 0.02
    python sequential_finetune.py --replay s2l --ratio 0.02
"""
import argparse
import json
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

import common as C

MAXLEN = 768


class SFTData(torch.utils.data.Dataset):
    def __init__(self, rows, tok):
        self.items = [C.encode(tok, r, MAXLEN) for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


@torch.no_grad()
def generate(model, tok, prompts, max_new, bs=16):
    model.eval()
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), bs):
        batch = [C.build_prompt(p) for p in prompts[i:i + bs]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in outs]


def eval_all(model, tok, tests):
    res = {}
    for t, rows in tests.items():
        preds = generate(model, tok, [r["prompt"] for r in rows], C.TASKS[t]["max_new"])
        res[t] = C.score(t, preds, [r["response"] for r in rows])
        print(f"    eval {t}: {res[t]:.3f}   e.g. pred={preds[0][:80]!r}")
    return res


def make_buffer(mode, task, rows, ratio, seed, root):
    if mode == "none":
        return []
    if mode == "full":
        return list(rows)
    if mode == "random":
        k = max(1, int(round(ratio * len(rows))))
        return random.Random(seed).sample(rows, k)
    if mode == "s2l":
        with open(f"{root}/buffers/s2l_{ratio}/{task}.json") as f:
            idx = json.load(f)
        assert max(idx) < len(rows), f"buffer index out of range for {task}: data/ and buffers/ are out of sync"
        return [rows[i] for i in idx]
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", default=",".join(C.ORDER))
    ap.add_argument("--replay", choices=["none", "random", "s2l", "full"], default="none")
    ap.add_argument("--ratio", type=float, default=0.02)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="50 train / 20 test examples, 1 epoch")
    args = ap.parse_args()

    order = args.order.split(",")
    R_ = C.root(args.debug)
    name = f"{args.replay}_s{args.seed}" if args.replay in ("none", "full") else f"{args.replay}_{args.ratio}_s{args.seed}"
    os.makedirs(f"{R_}/results/{name}", exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(C.TARGET_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        C.TARGET_MODEL, torch_dtype=torch.bfloat16 if use_bf16 else torch.float16, device_map="auto")
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    if not use_bf16:  # fp16 GPUs (e.g. Colab T4): keep trainable LoRA weights in fp32
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    model.print_trainable_parameters()

    train, test = {}, {}
    for t in order:
        train[t], test[t] = C.load_task(t, debug=args.debug)
        print(f"{t}: {len(train[t])} train / {len(test[t])} test")

    # LoRA B matrices start at zero, so this equals the zero-shot base model
    print("\n=== zero-shot base model ===")
    R = {"base": eval_all(model, tok, test)}

    for k, t in enumerate(order):
        rows = list(train[t])
        for prev in order[:k]:
            rows += make_buffer(args.replay, prev, train[prev], args.ratio, args.seed, R_)
        random.Random(args.seed).shuffle(rows)
        print(f"\n=== Stage {k + 1}/{len(order)}: {t} | {len(train[t])} new + {len(rows) - len(train[t])} replay rows ===")

        t0 = time.time()
        targs = TrainingArguments(
            output_dir=f"{R_}/tmp/{name}", per_device_train_batch_size=8, gradient_accumulation_steps=2,
            learning_rate=args.lr, num_train_epochs=1 if args.debug else args.epochs,
            lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=20, save_strategy="no",
            report_to="none", bf16=use_bf16, fp16=not use_bf16, remove_unused_columns=False, seed=args.seed)
        Trainer(model=model, args=targs, train_dataset=SFTData(rows, tok),
                data_collator=lambda b: C.collate(b, tok.pad_token_id)).train()
        print(f"  trained in {(time.time() - t0) / 60:.1f} min")

        model.save_pretrained(f"{R_}/adapters/{name}/after_{t}")
        R[t] = eval_all(model, tok, test)
        with open(f"{R_}/results/{name}/results.json", "w") as f:
            json.dump({"order": order, "replay": args.replay, "ratio": args.ratio,
                       "seed": args.seed, "R": R}, f, indent=2)

    print("\nResults matrix (rows = after training on, cols = evaluated on)")
    print("after".ljust(10) + "".join(c.ljust(10) for c in order))
    for r, v in R.items():
        print(r.ljust(10) + "".join(f"{v[c]:.3f}".ljust(10) for c in order))


if __name__ == "__main__":
    main()
