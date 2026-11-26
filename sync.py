"""
Exchange files between machines through a PRIVATE Hugging Face dataset repo.

Environment (set in the notebook):
  HF_TOKEN  - Hugging Face token with WRITE permission
  S2L_REPO  - 'username/repo-name' (same on every machine; created automatically, private)

Usage (paths are relative to the project folder; a path can be a file or a folder):
  python sync.py push data/gsm8k.json buffers trajectories
  python sync.py pull data buffers
  python sync.py wait data/gsm8k.json data/xsum.json [--timeout 12] [--poll 60]
      -> blocks until every listed file exists in the repo (timeout in hours)
"""
import argparse
import os
import sys
import time

from huggingface_hub import HfApi, snapshot_download


def _cfg():
    repo, tok = os.environ.get("S2L_REPO", ""), os.environ.get("HF_TOKEN", "")
    if "/" not in repo:
        sys.exit("Set S2L_REPO='username/repo-name' (same on every machine).")
    if not tok:
        sys.exit("Set HF_TOKEN (needs WRITE permission).")
    return repo, tok, HfApi(token=tok)


def _norm(p):
    return os.path.normpath(p).replace("\\", "/")


def _retry(fn, tries=4):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            print(f"  retry {i + 1}/{tries - 1}: {e}")
            time.sleep(5 * (i + 1))


def push(paths):
    repo, _, api = _cfg()
    _retry(lambda: api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True))
    for p in map(_norm, paths):
        if os.path.isdir(p):
            _retry(lambda p=p: api.upload_folder(folder_path=p, path_in_repo=p, repo_id=repo,
                                                 repo_type="dataset", commit_message=f"push {p}"))
        elif os.path.isfile(p):
            _retry(lambda p=p: api.upload_file(path_or_fileobj=p, path_in_repo=p, repo_id=repo,
                                               repo_type="dataset", commit_message=f"push {p}"))
        else:
            print(f"  skip (not found): {p}")
            continue
        print(f"  pushed {p}")


def pull(paths):
    repo, tok, _ = _cfg()
    pats = []
    for p in map(_norm, paths):
        pats += [p, p + "/*"]  # file or whole folder
    _retry(lambda: snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=pats,
                                     local_dir=".", token=tok))
    print(f"  pulled {', '.join(map(_norm, paths))}")


def wait(files, timeout_h, poll):
    repo, _, api = _cfg()
    files = [_norm(f) for f in files]
    end, last = time.time() + timeout_h * 3600, None
    while True:
        try:
            have = set(api.list_repo_files(repo, repo_type="dataset"))
        except Exception as e:  # repo may not exist yet
            have = set()
            print(f"  repo not readable yet ({type(e).__name__})")
        missing = [f for f in files if f not in have]
        if not missing:
            print("  all files present")
            return
        if time.time() > end:
            sys.exit(f"timed out waiting for: {missing}")
        if missing != last:
            print(f"{time.strftime('%H:%M:%S')} waiting for {missing}")
            last = missing
        time.sleep(poll)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("push", "pull"):
        sub.add_parser(c).add_argument("paths", nargs="+")
    w = sub.add_parser("wait")
    w.add_argument("files", nargs="+")
    w.add_argument("--timeout", type=float, default=12.0, help="hours")
    w.add_argument("--poll", type=int, default=60, help="seconds between checks")
    a = ap.parse_args()
    {"push": lambda: push(a.paths), "pull": lambda: pull(a.paths),
     "wait": lambda: wait(a.files, a.timeout, a.poll)}[a.cmd]()
