#!/usr/bin/env bash
# Run from the project folder. Do the DEBUG block first on a fresh machine (a few minutes).
# NOTE: if you change n_train/n_test in common.py, delete data/ trajectories/ buffers/ results/ adapters/ first.
set -e

# ---- 0. smoke test on 50 examples per task (writes to ./debug, never touches real results)
python select_s2l.py --debug --ratios 0.1
python sequential_finetune.py --debug --replay s2l --ratio 0.1
python aggregate_results.py --root debug

# ---- 1. selection with the Pythia-160M proxy (first run also caches data/*.json: commit it!)
#      ratio 0.02 -> 140 (GSM8K) / 160 (WikiSQL, XSum) rows; ratio 0.05 -> 350 / 400 rows
python select_s2l.py --ratios 0.02,0.05

# ---- 2. continual fine-tuning runs (change --seed for extra seeds)
python sequential_finetune.py --replay none
python sequential_finetune.py --replay random --ratio 0.02
python sequential_finetune.py --replay s2l    --ratio 0.02
# optional: upper bound and 5% buffers
# python sequential_finetune.py --replay full
# python sequential_finetune.py --replay random --ratio 0.05
# python sequential_finetune.py --replay s2l    --ratio 0.05

# ---- 3. tables + plot
python aggregate_results.py
