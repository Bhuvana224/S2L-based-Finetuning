"""
Resume sequential (continual) fine-tuning from a saved stage adapter.

sequential_finetune.py saves the LoRA adapter after every stage to
    <root>/adapters/<run name>/after_<task>/
Only that adapter carries state between stages (each stage builds a fresh Trainer/optimizer/LR
schedule), so loading it and continuing at the next stage reproduces an uninterrupted run, apart
from the random state (dropout / shuffle order), which cannot be restored.

This file does NOT modify sequential_finetune.py or any other script: it imports them and reuses
their functions, and mirrors sequential_finetune.main() for the training loop.

Use the SAME --replay / --ratio / --seed / --order / --epochs / --lr as the original run, so the
replay buffers and hyperparameters are identical. Random-replay buffers are deterministic from
(seed, train data), so they come out the same.

Usage (resume after GSM8K, i.e. start at stage 2; the stage is inferred from the folder name):
    python resume_finetune.py --resume_from adapters/random_0.02_s0/after_gsm8k \
        --replay random --ratio 0.02 --seed 0
On a 15 GB GPU (T4) keep the effective batch of 16 but use smaller pieces:
    ... --bs 2 --accum 8
"""
import argparse
import json
import os
import random
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

import common as C
import sequential_finetune as SF


def run_name(replay, ratio, seed):
    # same naming as sequential_finetune.main so outputs land in the same results/ and adapters/ dirs
    if replay in ("none", "full"):
        return f"{replay}_s{seed}"
    return f"{replay}_{ratio}_s{seed}"


def infer_start(resume_from, order):
    """Index (0-based) of the stage to train next, from a folder named after_<task>."""
    base = os.path.basename(os.path.normpath(resume_from))
    if not base.startswith("after_") or base[len("after_"):] not in order:
        raise SystemExit(f"Cannot infer the stage from '{base}' (expected after_<task> with <task> in "
                         f"{order}). Pass --start_stage explicitly.")
    return order.index(base[len("after_"):]) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume_from", required=True, help="saved adapter folder, e.g. adapters/random_0.02_s0/after_gsm8k")
    ap.add_argument("--start_stage", type=int, default=None,
                    help="1-indexed stage to train next (default: inferred from the --resume_from folder name)")
    ap.add_argument("--order", default=",".join(C.ORDER))
    ap.add_argument("--replay", choices=["none", "random", "s2l", "full"], default="none")
    ap.add_argument("--ratio", type=float, default=0.02)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8, help="per-device batch size (original: 8)")
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps (original: 2)")
    ap.add_argument("--prev_results", default=None,
                    help="results.json of the earlier run to copy the base / earlier rows from "
                         "(default: <root>/results/<name>/results.json if it exists)")
    ap.add_argument("--name", default=None, help="run name (default: same as the original run)")
    ap.add_argument("--debug", action="store_true", help="50 train / 20 test examples, 1 epoch")
    args = ap.parse_args()

    order = args.order.split(",")
    start = (args.start_stage - 1) if args.start_stage else infer_start(args.resume_from, order)
    if not 1 <= start < len(order):
        raise SystemExit(f"Nothing to resume: next stage index {start + 1} is outside 2..{len(order)}.")

    R_ = C.root(args.debug)
    name = args.name or run_name(args.replay, args.ratio, args.seed)
    os.makedirs(f"{R_}/results/{name}", exist_ok=True)

    # the adapter must match what sequential_finetune.py builds
    cfg_path = os.path.join(args.resume_from, "adapter_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    if cfg.get("base_model_name_or_path") != C.TARGET_MODEL or cfg.get("r") != 16:
        raise SystemExit(f"Adapter was not trained with {C.TARGET_MODEL} / r=16: "
                         f"{cfg.get('base_model_name_or_path')} / r={cfg.get('r')}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(C.TARGET_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        C.TARGET_MODEL, torch_dtype=torch.bfloat16 if use_bf16 else torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
    if not use_bf16:  # fp16 GPUs (e.g. Colab T4): keep trainable LoRA weights in fp32
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    model.print_trainable_parameters()

    train, test = {}, {}
    for t in order:
        train[t], test[t] = C.load_task(t, debug=args.debug)
        print(f"{t}: {len(train[t])} train / {len(test[t])} test")

    # ---- results matrix: rows already available + the loaded adapter's row
    prev_path = args.prev_results or f"{R_}/results/{name}/results.json"
    prev = {}
    if os.path.exists(prev_path):
        with open(prev_path) as f:
            prev = json.load(f).get("R", {})
        print(f"copying earlier rows from {prev_path}: {list(prev)}")

    R = {}
    if "base" in prev:
        R["base"] = prev["base"]
    else:
        print("\n=== zero-shot base model (adapter disabled) ===")
        with model.disable_adapter():
            R["base"] = SF.eval_all(model, tok, test)
    for t in order[:start - 1]:
        if t in prev:
            R[t] = prev[t]
        else:
            print(f"  WARNING: no saved scores for stage '{t}'; its row will be missing from the results matrix")
    last_done = order[start - 1]
    print(f"\n=== loaded adapter (after {last_done}): evaluating ===")
    R[last_done] = SF.eval_all(model, tok, test)

    def dump():
        with open(f"{R_}/results/{name}/results.json", "w") as f:
            json.dump({"order": order, "replay": args.replay, "ratio": args.ratio, "seed": args.seed,
                       "resumed_from": args.resume_from, "R": R}, f, indent=2)
    dump()

    for k in range(start, len(order)):
        t = order[k]
        rows = list(train[t])
        for p_ in order[:k]:
            rows += SF.make_buffer(args.replay, p_, train[p_], args.ratio, args.seed, R_)
        random.Random(args.seed).shuffle(rows)
        print(f"\n=== Stage {k + 1}/{len(order)}: {t} | {len(train[t])} new + {len(rows) - len(train[t])} replay rows ===")

        t0 = time.time()
        targs = TrainingArguments(
            output_dir=f"{R_}/tmp/{name}", per_device_train_batch_size=args.bs,
            gradient_accumulation_steps=args.accum,
            learning_rate=args.lr, num_train_epochs=1 if args.debug else args.epochs,
            lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=20, save_strategy="no",
            report_to="none", bf16=use_bf16, fp16=not use_bf16, remove_unused_columns=False, seed=args.seed)
        Trainer(model=model, args=targs, train_dataset=SF.SFTData(rows, tok),
                data_collator=lambda b: C.collate(b, tok.pad_token_id)).train()
        print(f"  trained in {(time.time() - t0) / 60:.1f} min")

        model.save_pretrained(f"{R_}/adapters/{name}/after_{t}")
        R[t] = SF.eval_all(model, tok, test)
        dump()

    print("\nResults matrix (rows = after training on, cols = evaluated on)")
    print("after".ljust(10) + "".join(c.ljust(10) for c in order))
    for r, v in R.items():
        print(r.ljust(10) + "".join(f"{v[c]:.3f}".ljust(10) for c in order))


if __name__ == "__main__":
    main()
