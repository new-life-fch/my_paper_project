#!/usr/bin/env python3
"""
Bulk-download every model the paper needs, into the DATA disk.

Why this script exists (2026-10-13, new AutoDL host):
  - The old GPU host was recycled; results/cache/ and the HF cache are gone.
  - This host has a 30G system disk (/) and a 150G data disk (/root/autodl-tmp).
    `~/.cache/huggingface` lands on the SYSTEM disk, so HF_HOME must be set.
  - Download channel: see scripts/setup_autodl.sh.  Measured on this host,
    huggingface.co through the AutoDL proxy stalls on multi-GB shards, while
    hf-mirror.com direct runs a steady ~5 MB/s.  Either way, transfers are
    flaky enough that a **stall watchdog** is required: a hung read does not
    raise, it just sits there (observed: 90s with zero byte growth).

Model list matches paper Table 1 (8 models, 3B-8B, LLaMA / Mistral / Qwen,
base + instruct).  The 4 LLaMA repos are gated=manual -> they need an HF token
whose account has accepted the Meta licence.

Usage:
    bash scripts/setup_autodl.sh                 # recommended: sets env + calls this
    # or by hand:
    export HF_HOME=/root/autodl-tmp/hf_cache
    export HF_ENDPOINT=https://hf-mirror.com
    export HF_TOKEN=hf_xxx
    python scripts/download_all_models.py [--only llama32_3b,qwen25_7b]

Never writes the token anywhere; huggingface_hub stores it under HF_HOME.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time

# (tag, repo_id, needs_token)
MODELS = [
    ("llama32_3b",          "meta-llama/Llama-3.2-3B",             True),
    ("llama32_3b_instruct", "meta-llama/Llama-3.2-3B-Instruct",    True),
    ("llama31_8b",          "meta-llama/Llama-3.1-8B",             True),
    ("llama31_8b_instruct", "meta-llama/Llama-3.1-8B-Instruct",    True),
    ("mistral_7b",          "mistralai/Mistral-7B-v0.3",           False),
    ("mistral_7b_instruct", "mistralai/Mistral-7B-Instruct-v0.3",  False),
    ("qwen25_7b",           "Qwen/Qwen2.5-7B",                     False),
    ("qwen25_7b_instruct",  "Qwen/Qwen2.5-7B-Instruct",            False),
]

# everything from_pretrained / AutoTokenizer needs, nothing else (no gguf/flax/original)
ALLOW = ["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"]

DEFAULT_HF_HOME = "/root/autodl-tmp/hf_cache"
MIRROR = "https://hf-mirror.com"


class Stalled(RuntimeError):
    """Raised by the watchdog when the cache stops growing mid-download."""


class StallWatchdog:
    """Abort a transfer that stops making progress.

    `snapshot_download` has no stall timeout of its own: a dead socket read
    just blocks forever (HF_HUB_DOWNLOAD_TIMEOUT only covers the initial
    response).  On timeout we raise inside the calling thread, and the caller
    retries -- snapshot_download resumes .incomplete blobs.

    Progress signal = the newest mtime under the cache.  Do NOT use total byte
    size: this cache holds multi-GB files, so summing them costs seconds per
    poll and a slow poll can exceed the stall window by itself (observed on
    the first version of this class).
    """

    def __init__(self, path, stall_secs=180, poll=10, max_depth=3):
        self.path = path
        self.stall_secs = stall_secs
        self.poll = poll
        self.max_depth = max_depth
        self._stop = threading.Event()
        self._thread = None
        self._main = None

    def _latest_mtime(self):
        base_depth = self.path.rstrip("/").count("/")
        newest = 0.0
        for root, dirs, files in os.walk(self.path):
            if root.count("/") - base_depth >= self.max_depth:
                dirs[:] = []
            for f in files:
                try:
                    m = os.stat(os.path.join(root, f)).st_mtime
                    if m > newest:
                        newest = m
                except OSError:
                    pass
        return newest

    def _run(self):
        last, last_change = self._latest_mtime(), time.time()
        while not self._stop.wait(self.poll):
            now = self._latest_mtime()
            if now > last:
                last, last_change = now, time.time()
            elif time.time() - last_change > self.stall_secs:
                import ctypes
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(self._main), ctypes.py_object(Stalled))
                return

    def __enter__(self):
        self._main = threading.get_ident()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return False


def du(path):
    try:
        out = subprocess.run(["du", "-sm", path], capture_output=True, text=True, check=True)
        return int(out.stdout.split()[0])
    except Exception:
        return -1


def free_gb(path):
    total, used, free = shutil.disk_usage(path)
    return free / 1024 ** 3


def check_env():
    hf_home = os.environ.get("HF_HOME", "")
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    ok = True
    if not hf_home:
        print("!! HF_HOME is unset -> the cache would land on the 30G system disk.")
        print(f"   export HF_HOME={DEFAULT_HF_HOME}")
        ok = False
    elif not hf_home.startswith("/root/autodl-tmp"):
        print(f"!! HF_HOME={hf_home} is NOT on the data disk (/root/autodl-tmp).")
        ok = False

    proxied = bool(os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY"))
    if endpoint.startswith(MIRROR):
        # measured: the AutoDL proxy makes hf-mirror stall completely
        if proxied:
            print("!! HF_ENDPOINT=hf-mirror but a proxy IS set -> stalls (measured).")
            print("   unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY")
            ok = False
    elif not proxied:
        print("!! huggingface.co without a proxy is unreachable from this host.")
        print("   source /etc/network_turbo   (or use HF_ENDPOINT=hf-mirror.com)")
        ok = False

    try:
        import hf_xet  # noqa: F401
        print("!! hf_xet is importable -> it bypasses classic HTTP and hangs here.")
        print("   python -m pip uninstall -y hf_xet")
        ok = False
    except ImportError:
        pass
    return ok


def cached_complete(repo_id):
    """True if this repo is already fully downloaded under HF_HOME.

    Checks the *content*, not just "a snapshot dir exists": every shard named in
    model.safetensors.index.json must be present and non-empty, and every snapshot
    symlink must resolve.  This is what lets a re-run skip models that are done --
    worth having because the HuggingFace cache stores each blob under a hash and a
    killed transfer can leave an orphaned `.incomplete` next to the real file
    (observed here: 3B-Instruct looked "in progress" but was complete).
    """
    from huggingface_hub import constants, try_to_load_from_cache
    cache_root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    d = os.path.join(cache_root, "models--" + repo_id.replace("/", "--"))
    if not os.path.isdir(d):
        # fall back to the library's own view (respects HF_HUB_CACHE)
        d = os.path.join(constants.HF_HUB_CACHE, "models--" + repo_id.replace("/", "--"))
        if not os.path.isdir(d):
            return False
    snaps = glob.glob(os.path.join(d, "snapshots", "*"))
    if not snaps:
        return False
    snap = snaps[0]

    idx = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(idx):
        try:
            with open(idx) as f:
                shards = sorted(set(json.load(f)["weight_map"].values()))
        except Exception:
            return False
        for sh in shards:
            p = os.path.join(snap, sh)
            if not os.path.exists(p) or os.path.getsize(p) < 1_000_000:
                return False
    else:
        # single-file model: any real weight file will do
        weights = [f for f in os.listdir(snap) if f.endswith((".safetensors", ".bin"))]
        if not weights or all(os.path.getsize(os.path.join(snap, f)) < 1_000_000 for f in weights):
            return False

    # every symlink must resolve to a real blob
    for f in os.listdir(snap):
        p = os.path.join(snap, f)
        if os.path.islink(p) and not os.path.exists(p):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated tags to restrict the run")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--stall-secs", type=int, default=180,
                    help="abort+retry a transfer that shows no byte growth this long")
    args = ap.parse_args()

    only = {t for t in args.only.split(",") if t}
    todo = [m for m in MODELS if not only or m[0] in only]

    print("=" * 72)
    print(" HF bulk model download")
    print("=" * 72)
    if not check_env():
        print("\nEnvironment not ready; fix the items above first.")
        return 2

    from huggingface_hub import login, snapshot_download

    tok = os.environ.get("HF_TOKEN")
    if tok:
        login(token=tok, add_to_git_credential=False)
        print("HF token: provided via $HF_TOKEN (logged in)")
    else:
        try:
            from huggingface_hub import HfFolder
            print("HF token: from existing login store ->", bool(HfFolder.get_token()))
        except Exception:
            print("HF token: unknown (gated LLaMA repos will 401 if absent)")

    cache_dir = os.path.join(os.environ["HF_HOME"], "hub")
    print(f"HF_HOME   : {os.environ['HF_HOME']}")
    print(f"endpoint  : {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
    print(f"free space: {free_gb(os.environ['HF_HOME']):.1f} GB on the data disk")
    print(f"models    : {len(todo)}\n")

    results = []
    for i, (tag, repo, gated) in enumerate(todo, 1):
        print("-" * 72, flush=True)
        print(f"[{i}/{len(todo)}] {tag}  <-  {repo}   (gated={gated})", flush=True)
        print("-" * 72, flush=True)
        if cached_complete(repo):
            print("  SKIP  already complete in the cache (all shards present)", flush=True)
            results.append({"tag": tag, "repo": repo, "ok": True, "error": "",
                            "seconds": 0.0, "cache_grew_mb": 0, "skipped": True})
            continue
        before = du(cache_dir) if os.path.isdir(cache_dir) else 0
        t0 = time.time()
        ok, err = False, ""
        for attempt in range(1, args.retries + 1):
            try:
                with StallWatchdog(cache_dir, stall_secs=args.stall_secs):
                    path = snapshot_download(
                        repo_id=repo,
                        allow_patterns=ALLOW,
                        max_workers=args.max_workers,
                    )
                ok = True
                break
            except Exception as e:                      # noqa: BLE001 - report and retry
                err = f"{type(e).__name__}: {e}"
                print(f"  attempt {attempt}/{args.retries} failed: {err[:200]}", flush=True)
                if attempt < args.retries:
                    # stale .lock files from a killed transfer make the next call hang
                    for root, _dirs, files in os.walk(cache_dir):
                        for f in files:
                            if f.endswith(".lock"):
                                try:
                                    os.remove(os.path.join(root, f))
                                except OSError:
                                    pass
                    time.sleep(10 * attempt)
        dt = time.time() - t0
        after = du(cache_dir) if os.path.isdir(cache_dir) else 0
        grew = after - before
        results.append({"tag": tag, "repo": repo, "ok": ok, "error": err,
                        "seconds": round(dt, 1), "cache_grew_mb": grew})

        if ok:
            n_files = sum(len(f) for _, _, f in os.walk(path))
            print(f"  OK  cached to {path}")
            print(f"      {n_files} files, cache grew {grew} MB, {dt/60:.1f} min, "
                  f"free now {free_gb(os.environ['HF_HOME']):.1f} GB", flush=True)
        else:
            print(f"  FAILED after {args.retries} attempts: {err[:200]}", flush=True)

    print("\n" + "=" * 72)
    print(" SUMMARY")
    print("=" * 72)
    for r in results:
        flag = "OK  " if r["ok"] else "FAIL"
        mb_per_s = (r["cache_grew_mb"] / r["seconds"]) if r["seconds"] else 0
        print(f"  {flag} {r['tag']:22s} {r['seconds']/60:6.1f} min  {r['cache_grew_mb']:6d} MB "
              f"({mb_per_s:5.1f} MB/s)  {r['repo']}")
        if not r["ok"]:
            print(f"       {r['error'][:200]}")
    print(f"\nfree space on data disk: {free_gb(os.environ['HF_HOME']):.1f} GB")
    n_ok = sum(r["ok"] for r in results)
    print(f"downloaded {n_ok}/{len(results)}")

    log = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "logs")
    os.makedirs(log, exist_ok=True)
    with open(os.path.join(log, "download_all_models.json"), "w") as f:
        json.dump({"results": results, "free_gb": free_gb(os.environ["HF_HOME"]),
                   "endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co")},
                  f, indent=2)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
