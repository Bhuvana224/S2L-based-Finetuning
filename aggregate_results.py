"""
Stage 3 - Aggregate results/<run>/results.json into tables and a plot.

Runs named like  s2l_0.05_s0, s2l_0.05_s1  are grouped (mean +- std over seeds) as  s2l_0.05.

Metrics (order = [T1..Tn], R[a][b] = score on task b after training through task a):
  avg final  = mean_j R[Tn][Tj]
  forgetting = mean over j<n of ( max_{j<=l<=n-1} R[Tl][Tj]  -  R[Tn][Tj] )   (lower is better)
  BWT        = mean over j<n of ( R[Tn][Tj] - R[Tj][Tj] )                     (higher is better)

Usage:  python aggregate_results.py [--root .]
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def metrics(R, order):
    T = len(order)
    last = R[order[-1]]
    avg = float(np.mean([last[t] for t in order]))
    fgt = float(np.mean([max(R[order[l]][order[j]] for l in range(j, T - 1)) - last[order[j]]
                         for j in range(T - 1)]))
    bwt = float(np.mean([last[order[j]] - R[order[j]][order[j]] for j in range(T - 1)]))
    return avg, fgt, bwt


def ms(xs):
    return f"{np.mean(xs):.3f} ± {np.std(xs):.3f}" if len(xs) > 1 else f"{xs[0]:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    groups, order, base = defaultdict(list), None, None
    for path in sorted(glob.glob(f"{args.root}/results/*/results.json")):
        d = json.load(open(path))
        if order is None:
            order, base = d["order"], d["R"]["base"]
        if d["order"] != order or d["order"][-1] not in d["R"]:
            print(f"skipping {path} (different task order or unfinished run)")
            continue
        method = re.sub(r"_s\d+$", "", os.path.basename(os.path.dirname(path)))
        groups[method].append(d["R"])
    if not groups:
        print("no finished runs found")
        return

    rows = []
    print(f"\nTask order: {' -> '.join(order)}   (zero-shot: " +
          ", ".join(f"{t} {base[t]:.3f}" for t in order) + ")\n")
    header = ["method", "seeds", "avg_final", "forgetting", "BWT"] + [f"final_{t}" for t in order]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for m, runs in groups.items():
        mt = np.array([metrics(R, order) for R in runs])
        finals = {t: [R[order[-1]][t] for R in runs] for t in order}
        row = [m, str(len(runs)), ms(mt[:, 0]), ms(mt[:, 1]), ms(mt[:, 2])] + [ms(finals[t]) for t in order]
        rows.append(row)
        print("| " + " | ".join(row) + " |")

    os.makedirs(f"{args.root}/results", exist_ok=True)
    with open(f"{args.root}/results/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    methods = list(groups)
    width = 0.8 / (len(methods) + 1)
    x = np.arange(len(order))
    plt.figure(figsize=(8, 4))
    plt.bar(x - 0.4 + width / 2, [base[t] for t in order], width, label="zero-shot", color="lightgray")
    for i, m in enumerate(methods):
        means = [np.mean([R[order[-1]][t] for R in groups[m]]) for t in order]
        errs = [np.std([R[order[-1]][t] for R in groups[m]]) for t in order]
        plt.bar(x - 0.4 + width * (i + 1.5), means, width, yerr=errs, label=m)
    plt.xticks(x, order)
    plt.ylabel("score after the final stage")
    plt.title("Performance on each task after training on all tasks")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{args.root}/results/final_scores.png", dpi=150)
    print(f"\nsaved {args.root}/results/summary.csv and final_scores.png")


if __name__ == "__main__":
    main()
