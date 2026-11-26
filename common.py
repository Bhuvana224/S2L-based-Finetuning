"""
Shared pieces used by every script: config, data loading, prompt format, encoding, metrics.

Tasks (in training order): GSM8K -> WikiSQL -> XSum -> PubMedQA
Target model: Qwen2.5-0.5B (LoRA).  Proxy model for S2L: Pythia-160M.

IMPORTANT: the first call to load_task() caches the selected rows to data/<task>.json.
Buffer files store INDICES into those cached rows, so every teammate must use the same
data/ folder (commit it to git) or the indices will point at the wrong examples.
"""
import json
import os
import re

TARGET_MODEL = "Qwen/Qwen2.5-0.5B"
PROXY_MODEL = "EleutherAI/pythia-160m"
SEED = 0                    # fixes the data subset; run seeds only change training randomness
DATA_DIR = "data"
ORDER = ["gsm8k", "wikisql", "xsum", "pubmedqa"]
MAX_CHARS = 1500            # truncate long inputs (XSum articles, PubMedQA abstracts)

TASKS = {
    "gsm8k":    dict(n_train=7000, n_test=300, max_new=256),   # full GSM8K train is ~7.47k
    "wikisql":  dict(n_train=8000, n_test=300, max_new=96),
    "xsum":     dict(n_train=8000, n_test=300, max_new=64),
    # PubMedQA's labelled set has only 1,000 examples (yes/no/maybe): split 700 train / 300 test
    "pubmedqa": dict(n_train=700,  n_test=300, max_new=6),
}


def root(debug):
    """Debug runs write to ./debug so they never overwrite real results or buffers."""
    return "debug" if debug else "."


# ------------------------------------------------------------------ formatting
def _fmt_gsm8k(x):
    return ("Solve the math problem step by step. End with '#### <answer>'.\n\n" + x["question"],
            x["answer"])


def _fmt_wikisql(x):
    t = x["table"]
    if isinstance(t, str):
        t = json.loads(t)
    s = x["sql"]
    if isinstance(s, str):
        s = json.loads(s)
    cols = " | ".join(t["header"])
    return (f"Write a SQL query that answers the question.\n\nTable columns: {cols}\n\n"
            f"Question: {x['question']}", s["human_readable"])


def _fmt_xsum(x):
    return ("Summarize the article in one sentence.\n\n" + x["document"][:MAX_CHARS],
            x["summary"].strip())


def _fmt_pubmedqa(x):
    ctx = x["context"]
    if isinstance(ctx, dict):
        ctx = ctx["contexts"]
    abstract = " ".join(ctx)[:MAX_CHARS]
    return (f"Answer the question with yes, no, or maybe, using the abstract.\n\n"
            f"Abstract: {abstract}\n\nQuestion: {x['question']}", x["final_decision"])


FMT = {"gsm8k": _fmt_gsm8k, "wikisql": _fmt_wikisql, "xsum": _fmt_xsum, "pubmedqa": _fmt_pubmedqa}


# ------------------------------------------------------------------ loading
def _load_wikisql():
    from datasets import load_dataset
    errors = []
    # WikiSQL on the Hub uses a loading script. That needs datasets<4 and trust_remote_code.
    try:
        return load_dataset("Salesforce/wikisql", trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        errors.append(f"script load: {e}")
    # Fallback: the auto-converted parquet branch (not verified for this dataset's schema)
    try:
        return load_dataset("Salesforce/wikisql", revision="refs/convert/parquet")
    except Exception as e:  # noqa: BLE001
        errors.append(f"parquet branch: {e}")
    raise RuntimeError(
        "Could not load Salesforce/wikisql. Try: pip install 'datasets>=2.19,<4.0' "
        "(script-based datasets are not supported in datasets>=4).\n" + "\n".join(errors))


def _load_raw(name):
    from datasets import load_dataset
    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main")
        return ds["train"], ds["test"]
    if name == "wikisql":
        ds = _load_wikisql()
        return ds["train"], ds["test"]
    if name == "xsum":
        ds = load_dataset("EdinburghNLP/xsum")
        return ds["train"], ds["test"]
    if name == "pubmedqa":
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]  # 1,000 rows, one split
        sp = ds.train_test_split(test_size=TASKS[name]["n_test"], seed=SEED)
        return sp["train"], sp["test"]
    raise ValueError(name)


def load_task(name, debug=False):
    """Returns (train_rows, test_rows); each row is {"prompt": str, "response": str}."""
    path = os.path.join(DATA_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
    else:
        cfg = TASKS[name]
        tr, te = _load_raw(name)
        tr = tr.shuffle(seed=SEED).select(range(min(cfg["n_train"], len(tr))))
        te = te.shuffle(seed=SEED).select(range(min(cfg["n_test"], len(te))))

        def to_rows(ds):
            return [dict(zip(("prompt", "response"), FMT[name](x))) for x in ds]

        d = {"train": to_rows(tr), "test": to_rows(te)}
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump(d, f)
    if debug:  # tiny slice for pipeline smoke tests
        d = {"train": d["train"][:50], "test": d["test"][:20]}
    return d["train"], d["test"]


# ------------------------------------------------------------------ prompt + encoding
def build_prompt(p):
    return f"### Instruction:\n{p}\n\n### Response:\n"


def encode(tok, row, maxlen):
    """Token ids + labels. Loss is computed on the response only (prompt tokens are -100)."""
    p = tok(build_prompt(row["prompt"]), add_special_tokens=False).input_ids
    r = tok(row["response"], add_special_tokens=False).input_ids + [tok.eos_token_id]
    return {"input_ids": (p + r)[:maxlen], "labels": ([-100] * len(p) + r)[:maxlen]}


def collate(batch, pad_id):
    import torch
    m = max(len(x["input_ids"]) for x in batch)
    ids = torch.tensor([x["input_ids"] + [pad_id] * (m - len(x["input_ids"])) for x in batch])
    lab = torch.tensor([x["labels"] + [-100] * (m - len(x["labels"])) for x in batch])
    att = torch.tensor([[1] * len(x["input_ids"]) + [0] * (m - len(x["input_ids"])) for x in batch])
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


# ------------------------------------------------------------------ metrics
def clean(pred):
    # Cut off if the model starts a new "### Instruction:" block. Must not match the "####"
    # answer marker used by GSM8K, so require exactly three '#' followed by whitespace.
    return re.split(r"\n###\s", "\n" + pred.strip())[0].strip()


def gsm_ans(s):
    m = re.findall(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", s)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def _norm_sql(s):
    return re.sub(r"\s+", " ", s.lower().replace('"', "'").strip().rstrip(";")).strip()


def score(task, preds, refs):
    """gsm8k/wikisql/pubmedqa -> accuracy; xsum -> ROUGE-L F1. All in [0, 1]."""
    preds = [clean(p) for p in preds]
    n = len(refs)
    if task == "gsm8k":
        ok = 0
        for p, r in zip(preds, refs):
            a, b = gsm_ans(p), gsm_ans(r)
            ok += a is not None and b is not None and abs(a - b) < 1e-6
        return ok / n
    if task == "wikisql":
        return sum(_norm_sql(p) == _norm_sql(r) for p, r in zip(preds, refs)) / n
    if task == "xsum":
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rougeL"])
        return sum(sc.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)) / n
    if task == "pubmedqa":
        def first(s):
            w = re.findall(r"[a-z]+", s.lower())
            return w[0] if w else ""
        return sum(first(p) == first(r) for p, r in zip(preds, refs)) / n
    raise ValueError(task)
