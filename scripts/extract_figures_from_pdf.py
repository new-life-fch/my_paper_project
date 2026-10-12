#!/usr/bin/env python3
"""
Recover the paper figures that were lost from results/figures/ (the activation
caches needed to re-render them are gone, and no GPU is available).

paper/main.pdf was compiled while both figures still existed, so the original
PNGs are still embedded in it as image XObjects (RGB + grayscale /SMask alpha).
This script walks the PDF's image XObjects and re-writes them as lossless PNGs,
merging the soft mask back in as an alpha channel.

Usage:
  python scripts/extract_figures_from_pdf.py --pdf paper/main.pdf \
      --outdir results/figures [--dry-run]
"""
import argparse
import os
import re

from pypdf import PdfReader

# XObject size -> paper figure name.  Sizes come from the source scripts:
#   plot_layer_signal.py: figsize (21, 6) @ dpi=150, bbox_inches="tight"
#   plot_causal.py:       figsize (11, 4.2) @ dpi=150, bbox_inches="tight"
SIZE_TO_NAME = {
    (3133, 887): "llama32_3b_layer_signal.png",
    (1634, 609): "causal_steer_3b.png",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="paper/main.pdf")
    ap.add_argument("--outdir", default="results/figures")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    reader = PdfReader(args.pdf)
    os.makedirs(args.outdir, exist_ok=True)

    seen = {}
    for pno, page in enumerate(reader.pages, 1):
        for img in page.images:
            pil = img.image  # pypdf already merges /SMask into an alpha channel
            key = (pil.width, pil.height)
            seen.setdefault(key, []).append((pno, img.name, pil.mode))

    print(f"image XObjects found in {args.pdf}:")
    for (w, h), occ in sorted(seen.items()):
        pages = sorted({p for p, _, _ in occ})
        modes = sorted({m for _, _, m in occ})
        name = SIZE_TO_NAME.get((w, h), "<unmapped>")
        print(f"  {w}x{h} mode={modes} pages={pages} -> {name}")

    if args.dry_run:
        return

    written = []
    for pno, page in enumerate(reader.pages, 1):
        for img in page.images:
            pil = img.image
            key = (pil.width, pil.height)
            if key not in SIZE_TO_NAME:
                print(f"  ! skipping unmapped image {key} ({img.name})")
                continue
            dest = os.path.join(args.outdir, SIZE_TO_NAME[key])
            if os.path.exists(dest):
                continue  # don't clobber anything already present
            if pil.mode not in ("RGBA", "RGB", "L"):
                pil = pil.convert("RGBA")
            pil.save(dest, format="PNG", optimize=True)
            written.append((dest, pil.size, pil.mode, os.path.getsize(dest)))
    for dest, size, mode, nbytes in written:
        print(f"wrote {dest}  {size} {mode}  {nbytes/1024:.0f} KB")
    if not written:
        print("nothing written (all targets already exist)")


if __name__ == "__main__":
    main()
