"""
Record what produced a result, so runs on DIFFERENT hosts are comparable.

Why this exists: the original GPU host was recycled, so every number in
results/ will eventually be re-derived on a new machine.  The paper's AUCs
depend on library versions (sklearn's lbfgs solution path and numpy's BLAS
both move the 3rd decimal) and on the exact HF model revision.  None of that
was recorded anywhere -- requirements.txt pinned zero versions and meta.json
stored only the model *name*.  Without it, a fresh agent re-running the
pipeline sees "the numbers don't match" and cannot tell whether that is a bug,
a version difference, or a real finding.

Everything here is best-effort: a missing package must never break a run.
"""
import sys

_PKGS = ("torch", "transformers", "numpy", "sklearn", "scipy",
         "datasets", "nnsight", "joblib")


def env_versions():
    """{package: version-or-None} plus the Python version. Never raises."""
    out = {"python": sys.version.split()[0]}
    for mod in _PKGS:
        try:
            out[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            out[mod] = None
    return out


def model_revision(model):
    """HF commit hash of a loaded model, if the config carries one.

    transformers stamps `_commit_hash` on the config when it resolves a repo,
    so this works offline (unlike model_info(), which needs the Hub).
    """
    cfg = getattr(model, "config", None)
    return getattr(cfg, "_commit_hash", None) if cfg is not None else None


def describe_missing_pins():
    """One-line warning if requirements.txt pins no versions (it currently doesn't)."""
    try:
        import os
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "requirements.txt")
        with open(p, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        pinned = [l for l in lines if "==" in l]
        if not pinned:
            return ("requirements.txt pins no versions -> a new host may resolve "
                    "different sklearn/numpy and shift AUCs in the 3rd decimal")
    except Exception:
        pass
    return None
