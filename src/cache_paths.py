"""
Canonical cache-directory naming + legacy aliases.  See docs/CACHE_NAMING.md.

Canonical layout (set 2026-10-12):
    results/cache/{tag}_q{N}_{instruct,judge}          N in {500, 1500}
    results/cache/{tag}_q{N}_judge_fs{K}               few-shot judge variant

Before that rename, the 500-query caches were named inconsistently across
models -- LLaMA-3.2-3B (base) used the bare ``q500``, every other model used
``{tag}``, and instruct *models* produced ``{tag}_instruct_instruct``.  Those
old names still appear all over the historical documents (HANDOVER.md §2.4,
REPORT.md §7, the CLAUDE.md history sections) and, more importantly, **the
caches on the GPU host have probably not been renamed yet**.  So a reader that
insists on the canonical name would fail to find data that is right there.

``resolve_cache()`` therefore falls back to the legacy name and prints the
``mv`` command needed to migrate.  Aliases are listed EXPLICITLY rather than
derived, because derivation is unsafe: for a 1500-query cache the generic
"drop the _q1500" rule would land on the *500-query* directory of the same
model, i.e. silently the wrong dataset size.  An explicit table cannot do that.
"""
import os

#: canonical basename -> legacy basename(s) that may hold the same cache.
#: Only 500q MS MARCO caches and the FiQA cache were ever named differently;
#: the 1500q MS MARCO names were already canonical, so they appear nowhere here.
LEGACY_ALIASES = {
    # LLaMA-3.2-3B (base) 500q -- 'q500' was the original session's naming,
    # and run_model.sh later produced 'llama32_3b_instruct' for the same thing.
    "llama32_3b_q500_instruct": ["q500_instruct", "llama32_3b_instruct"],
    "llama32_3b_q500_judge":    ["q500_judge",    "llama32_3b_judge"],

    "llama31_8b_q500_instruct": ["llama31_8b_instruct"],
    "llama31_8b_q500_judge":    ["llama31_8b_judge"],

    "mistral_7b_q500_instruct": ["mistral_7b_instruct"],
    "mistral_7b_q500_judge":    ["mistral_7b_judge"],

    # instruct *models*: run_model.sh appended '_instruct' to a tag that
    # already ended in '_instruct'.
    "llama32_3b_instruct_q500_instruct": ["llama32_3b_instruct_instruct"],
    "llama32_3b_instruct_q500_judge":    ["llama32_3b_instruct_judge"],
    "llama31_8b_instruct_q500_instruct": ["llama31_8b_instruct_instruct"],
    "llama31_8b_instruct_q500_judge":    ["llama31_8b_instruct_judge"],
    "mistral_7b_instruct_q500_instruct": ["mistral_7b_instruct_instruct"],
    "mistral_7b_instruct_q500_judge":    ["mistral_7b_instruct_judge"],

    # FiQA ran at 1500q but the dir carried no query count.
    "llama32_3b_fiqa_q1500_instruct": ["llama32_3b_fiqa_instruct"],
    "llama32_3b_fiqa_q1500_judge":    ["llama32_3b_fiqa_judge"],
}


def legacy_aliases(cache_dir):
    """Legacy paths that may hold the cache asked for by `cache_dir`."""
    d, name = os.path.split(os.path.normpath(cache_dir))
    return [os.path.join(d, alt) for alt in LEGACY_ALIASES.get(name, [])]


def resolve_cache(cache_dir, verbose=True):
    """Return a directory that actually exists for a cache the caller wants.

    Prefers the canonical name; falls back to a known legacy name and tells the
    user how to migrate. Returns `cache_dir` unchanged when nothing is found, so
    downstream error messages still show the canonical path.
    """
    if os.path.isdir(cache_dir):
        return cache_dir
    for alt in legacy_aliases(cache_dir):
        if os.path.isdir(alt):
            if verbose:
                print(f"  ! cache not found at canonical path: {cache_dir}")
                print(f"    using LEGACY directory instead:    {alt}")
                print(f"    migrate it with:  mv {alt} {cache_dir}")
                print(f"    (or run scripts/migrate_cache_names.sh --apply)")
            return alt
    return cache_dir


def warn_if_legacy_exists(cache_dir, verbose=True):
    """For WRITERS: warn when a legacy directory already holds this cache.

    Without this, running e.g. extract_instruct.py with the new defaults on a
    host whose caches are still named the old way would create a *second*,
    multi-GB copy instead of reusing the existing one.
    """
    for alt in legacy_aliases(cache_dir):
        if os.path.isdir(alt):
            if verbose:
                print(f"  ! {os.path.dirname(cache_dir)} already contains a legacy "
                      f"directory for this cache: {alt}")
                print(f"    writing to {cache_dir} will create a SECOND copy.")
                print(f"    prefer:  mv {alt} {cache_dir}")
                print(f"    (or run scripts/migrate_cache_names.sh --apply)")
            return alt
    return None
