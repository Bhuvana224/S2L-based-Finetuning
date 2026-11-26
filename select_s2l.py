"""
Stage 1 - S2L subset selection with a SMALL PROXY model (Pythia-160M).

For each task, independently:
  1. Fine-tune a fresh proxy on the task's training rows.
  2. At several checkpoints during training, record the loss of EVERY training example
     -> each example gets a "loss trajectory" vector.
  3. Cluster the trajectories with k-means.
  4. Sample the same number of examples from each cluster (small clusters are taken whole,
     leftover budget goes to larger clusters) -> the replay buffer.
  5. Save the chosen indices to buffers/s2l_<ratio>/<task>.json

The Qwen target model is NOT used here.

Usage:
  python select_s2l.py --ratios 0.02,0.05   # ~140-160 / ~350-400 rows per task
  python select_s2l.py --debug            # tiny smoke test, writes to ./debug
  python select_s2l.py --reuse            # reuse saved trajectories, only re-cluster
"""
import argparse
import contextlib
import gc
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

import common as C

MAXLEN_PROXY = 512


def amp_ctx(device, use_amp):
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def per_example_loss(model, data, pad_id, device, use_amp, bs=8):
    """Mean token loss (response tokens only) for every example, in dataset order."""
    model.eval()
    out = []
    for i in range(0, len(data), bs):
        b = {k: v.to(device) for k, v in C.collate(data[i:i + bs], pad_id).items()}
        with amp_ctx(device, use_amp):
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        logits = logits.float()[:, :-1]
        lab = b["labels"][:, 1:]
        tok_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), lab.reshape(-1),
                                   ignore_index=-100, reduction="none").view(lab.shape)
        mask = (lab != -100).float()
        out += ((tok_loss * mask).sum(1) / mask.sum(1).clamp(min=1)).tolist()
    model.train()
    return out


def train_and_record(model, tok, rows, epochs, bs, lr, n_ckpts, device, use_amp, seed):
    """Fine-tune the proxy; return an (N, n_ckpts) array of per-example losses."""
    data = [C.encode(tok, r, MAXLEN_PROXY) for r in rows]
    n = len(data)
    total = epochs * ((n + bs - 1) // bs)
    ckpt_steps = set(np.unique(np.linspace(total / n_ckpts, total, n_ckpts).round().astype(int)).tolist())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    traj, step = [], 0
    model.train()
    for ep in range(epochs):
        order = list(range(n))
        rng.shuffle(order)
        for i in range(0, n, bs):
            b = {k: v.to(device) for k, v in C.collate([data[j] for j in order[i:i + bs]], tok.pad_token_id).items()}
            with amp_ctx(device, use_amp):
                loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step in ckpt_steps:
                traj.append(per_example_loss(model, data, tok.pad_token_id, device, use_amp))
                print(f"    checkpoint {len(traj)}/{len(ckpt_steps)} at step {step}/{total} "
                      f"(train loss {loss.item():.3f}, mean example loss {np.mean(traj[-1]):.3f})")
    return np.array(traj).T


def balanced_sample(labels, m, rng):
    """Pick m indices, spreading the budget evenly across clusters (small clusters taken whole)."""
    clusters = {c: [int(i) for i in np.where(labels == c)[0]] for c in sorted(set(labels.tolist()))}
    for c in clusters:
        rng.shuffle(clusters[c])
    order = sorted(clusters, key=lambda c: len(clusters[c]))
    chosen, remaining = [], m
    for i, c in enumerate(order):
        take = min(len(clusters[c]), remaining // (len(order) - i))
        chosen += clusters[c][:take]
        remaining -= take
    if remaining > 0:  # only if integer division left a remainder
        left = [i for i in range(len(labels)) if i not in set(chosen)]
        chosen += rng.sample(left, remaining)
    return sorted(chosen)


def select(traj, ratio, k, seed):
    from sklearn.cluster import KMeans
    n = len(traj)
    m = max(1, int(round(ratio * n)))
    k = max(1, min(k, m))
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(traj)
    return balanced_sample(labels, m, random.Random(seed)), labels


def plot_clusters(traj, labels, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.figure(figsize=(5, 3.5))
    for c in sorted(set(labels.tolist())):
        plt.plot(traj[labels == c].mean(0), alpha=0.8)
    plt.xlabel("checkpoint")
    plt.ylabel("mean example loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(C.ORDER[:-1]),
                    help="default skips the LAST task: nothing replays it, so it needs no buffer")
    ap.add_argument("--ratios", default="0.02")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--ckpts", type=int, default=8)
    ap.add_argument("--k", type=int, default=50, help="number of k-means clusters (capped at buffer size)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse", action="store_true", help="reuse saved trajectories")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    R = C.root(args.debug)
    ratios = [float(x) for x in args.ratios.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    os.makedirs(f"{R}/trajectories", exist_ok=True)
    os.makedirs(f"{R}/diagnostics", exist_ok=True)

    for t in args.tasks.split(","):
        print(f"\n=== {t} ===")
        rows, _ = C.load_task(t, debug=args.debug)
        tpath = f"{R}/trajectories/{t}.npy"
        if args.reuse and os.path.exists(tpath):
            traj = np.load(tpath)
            print(f"  loaded trajectories {traj.shape}")
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(C.PROXY_MODEL)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(C.PROXY_MODEL, torch_dtype=torch.float32).to(device)
            traj = train_and_record(model, tok, rows, args.epochs, args.bs, args.lr,
                                    args.ckpts, device, use_amp, args.seed)
            np.save(tpath, traj)
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        print(f"  trajectories {traj.shape}; mean loss first->last checkpoint: "
              f"{traj[:, 0].mean():.3f} -> {traj[:, -1].mean():.3f}")
        if np.abs(traj[:, 0].mean() - traj[:, -1].mean()) < 0.05:
            print("  WARNING: trajectories are nearly flat - the proxy barely learned this task; "
                  "clusters may carry little information.")

        for r in ratios:
            idx, labels = select(traj, r, args.k, args.seed)
            out_dir = f"{R}/buffers/s2l_{r}"
            os.makedirs(out_dir, exist_ok=True)
            with open(f"{out_dir}/{t}.json", "w") as f:
                json.dump(idx, f)
            sizes = np.bincount(labels).tolist()
            print(f"  ratio {r}: selected {len(idx)}/{len(rows)} examples; cluster sizes {sizes}")
            plot_clusters(traj, labels, f"{R}/diagnostics/{t}_{r}.png", f"{t} (ratio {r})")


if __name__ == "__main__":
    main()
