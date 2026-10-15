#!/usr/bin/env python3
"""
Repo self-check — run this before trusting the project, and after any edit.

Stdlib only (no torch/sklearn needed), so it works on a fresh clone, on a
laptop, and in CI.  Every check below corresponds to a mistake that was
ACTUALLY MADE in this repository at least once; the point is that a new agent
should be able to verify the invariants instead of rediscovering the breakage.

  python scripts/selfcheck.py          # exit 0 = all good
"""
import ast
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

CACHE_FNS = ("resolve_cache", "warn_if_legacy_exists")
# legacy cache dir names -- must appear ONLY in the alias table (and docs)
LEGACY_RE = re.compile(
    r'(?<![\w])((?:q500)|(?:llama31_8b|mistral_7b|llama32_3b_instruct|'
    r'llama31_8b_instruct|mistral_7b_instruct|llama32_3b_fiqa))_(instruct|judge)\b(?!_q\d)')

fails, warns = [], []


def ok(msg):
    print(f"  \033[32mOK\033[0m   {msg}")


def bad(msg):
    print(f"  \033[31mFAIL\033[0m {msg}")
    fails.append(msg)


def warn(msg):
    print(f"  \033[33mWARN\033[0m {msg}")
    warns.append(msg)


# ---------------------------------------------------------------- 1. imports
print("\n[1] every cache_paths call has a matching import (AST, not grep)")
for f in sorted(glob.glob("scripts/*.py")):
    tree = ast.parse(open(f, encoding="utf-8").read(), f)
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and "cache_paths" in n.module:
            imported |= {a.name for a in n.names}
    used = {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id in CACHE_FNS}
    missing = used - imported
    if missing:
        bad(f"{f} calls {sorted(missing)} without importing it -> NameError at runtime")
if not fails:
    ok("no un-imported cache_paths calls")

# ------------------------------------------------- 2. every reader resolves
print("\n[2] every script that READS a cache honours legacy names")
readers = []
for f in sorted(glob.glob("scripts/*.py")):
    src = open(f, encoding="utf-8").read()
    if re.search(r'"--(cache|judge-cache)"', src):
        readers.append(f)
        if not re.search(r'resolve_cache\(', src):
            bad(f"{f} takes a cache path but never calls resolve_cache()")
        # a WRITER is allowed to only warn; a reader must resolve
if readers and not any("resolve_cache" not in open(f, encoding="utf-8").read()
                       for f in readers):
    ok(f"{len(readers)} cache-taking scripts all resolve legacy names")

# --------------------------------------------------- 3. legacy names in code
print("\n[3] legacy cache names appear only in the alias table")
allowed = {os.path.join("src", "cache_paths.py")}
for f in sorted(glob.glob("scripts/*") + glob.glob("src/*.py")):
    if not os.path.isfile(f) or f in allowed:
        continue
    for i, line in enumerate(open(f, encoding="utf-8", errors="replace"), 1):
        low = line.lower()
        if "legacy" in low or "旧名" in line or "pre-2026" in low:
            continue                       # documentation of the old scheme
        m = LEGACY_RE.search(line)
        if m and not re.match(r'^\s*#', line):
            # model TAGS legitimately contain these substrings (e.g. a registry
            # mapping tag -> display name); only flag cache-path-looking lines
            if "cache" in low:
                bad(f"{f}:{i} legacy cache name in code: {m.group(0)!r}")
if not any("legacy cache name in code" in m for m in fails):
    ok("no stray legacy cache names in code paths")

# ------------------------------------------------------ 4. TODO references
print("\n[4] every S/P/R reference resolves to a real TODO item")
# TODO.md was archived to archive/TODO.md on 2026-10-15. The S/P/R ids are still
# cited from script headers and comments, so keep resolving them against the
# archived copy rather than dropping the check.
todo_candidates = ["TODO.md", "archive/TODO.md"]
todo_path = next((p for p in todo_candidates if os.path.exists(p)), None)
if todo_path:
    todo = open(todo_path, encoding="utf-8").read()
    defined = set(re.findall(r'^#{2,3} ([SPR]\d+-\d+)\.', todo, re.M))
    docs = ["README.md", "STATUS.md", "TODO.md", "archive/TODO.md",
            "archive/CLAUDE.md", "archive/HANDOVER.md",
            "archive/READING_GUIDE.md", "archive/CCFB_READINESS.md",
            "archive/HANDOVER_20261014_STOP.md", "archive/E11_JUDGEMENT.md",
            "docs/READING_GUIDE.md", "docs/CACHE_NAMING.md",
            "results/figures/README.md", "results/cache/README.md",
            ".claude/skills/write-paper/SKILL.md"]
    dangling = []
    for d in docs:
        if not os.path.exists(d):
            continue
        for m in sorted(set(re.findall(r'\b([SPR]\d+-\d+)\b',
                                       open(d, encoding="utf-8").read()))):
            if m not in defined:
                dangling.append(f"{d}: {m}")
    if dangling:
        for x in dangling:
            bad(f"dangling TODO reference -> {x}")
    else:
        ok(f"{len(defined)} items (from {todo_path}), 0 dangling references")
else:
    warn("TODO.md not found (looked in " + ", ".join(todo_candidates) + ")")

# ------------------------------------------------- 5. alias table soundness
print("\n[5] alias table never maps a 1500q cache onto a 500q sibling")
sys.path.insert(0, ROOT)
try:
    from src.cache_paths import LEGACY_ALIASES, legacy_aliases
    for canon in LEGACY_ALIASES:
        if "_q1500" in canon:
            for alt in LEGACY_ALIASES[canon]:
                if "_q500" in alt:
                    bad(f"{canon} aliases onto {alt} -- WRONG DATASET SIZE")
    # the generic "drop the _qN" rule must never apply to 1500q caches
    for probe in ("results/cache/llama32_3b_q1500_instruct",
                  "results/cache/qwen25_7b_q1500_judge",
                  "results/cache/llama31_8b_q1500_instruct"):
        if legacy_aliases(probe):
            bad(f"{probe} resolves to {legacy_aliases(probe)} -- 1500q must have no alias")
    ok("alias table sound (1500q caches have no legacy alias)")
except Exception as e:
    bad(f"could not check alias table: {e}")

# ---------------------------------------------- 6. prompts byte-identical
print("\n[6] the instruct prompt evaluates to the same string in every script")
# Compare the EVALUATED string, not the source text -- comparing source gives a
# false positive on comment/formatting differences, and a spurious warning is
# itself a divergence source (a new agent goes and investigates a non-issue).
try:
    PROBE_Q, PROBE_P = "QUERY", "PASSAGE"
    EXPECT = (f"Document: {PROBE_P}\nQuestion: {PROBE_Q}\n"
              f"Is the document relevant to the question? Answer:")
    seen = {}
    for f in sorted(glob.glob("scripts/*.py")):
        tree = ast.parse(open(f, encoding="utf-8").read(), f)
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and "prompt" in n.name.lower()]:
            ns = {}
            try:
                exec(compile(ast.Module(body=[fn], type_ignores=[]), f, "exec"), ns)
                got = ns[fn.name](PROBE_Q, PROBE_P)
            except Exception:
                continue
            if isinstance(got, str) and "Is the document relevant" in got:
                seen.setdefault(got, []).append(f"{os.path.basename(f)}:{fn.name}")
    if not seen:
        warn("no prompt builder could be evaluated -- check this manually")
    elif len(seen) == 1 and EXPECT in seen:
        ok(f"identical in all {sum(len(v) for v in seen.values())} builder(s): "
           f"{', '.join(sum(seen.values(), []))}")
    else:
        for got, where in seen.items():
            bad(f"prompt differs in {where}: {got!r}")
except Exception as e:
    warn(f"could not compare prompts: {e}")

# ------------------------------------------------------------------ summary
print("\n" + "=" * 62)
if fails:
    print(f"\033[31m{len(fails)} CHECK(S) FAILED\033[0m")
    for m in fails:
        print(f"  - {m}")
    sys.exit(1)
print(f"\033[32mALL CHECKS PASSED\033[0m" + (f"  ({len(warns)} warning(s))" if warns else ""))
