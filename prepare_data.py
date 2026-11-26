"""
Download + cache selected rows for the given tasks into data/<task>.json (same code path as every
other script, so the rows are identical). Used by the multi-machine notebooks.

Usage: python prepare_data.py --tasks gsm8k
"""
import argparse

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default=",".join(C.ORDER))
for t in ap.parse_args().tasks.split(","):
    tr, te = C.load_task(t)
    print(f"{t}: cached {len(tr)} train / {len(te)} test rows -> data/{t}.json")
