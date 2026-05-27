#!/usr/bin/env python3
"""Regenerate web favicons from repo-root icon.png (transparent background).

Removes the dark panel background via edge flood-fill, then writes:
  web/app/icon.png (1024), apple-icon.png (180), favicon.ico
  web/public/icon-192.png, icon-512.png

Usage:
  python scripts/generate_icons.py
  python scripts/generate_icons.py --source path/to/icon.png
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

from PIL import Image, ImageFilter

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _luminance(r: int, g: int, b: int) -> float:
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _is_background(r: int, g: int, b: int, a: int) -> bool:
    if a < 10:
        return True
    # Dark charcoal panels + subtle vertical stripes in the source art.
    return _luminance(r, g, b) < 48 and max(r, g, b) - min(r, g, b) < 35


def make_transparent(src: Image.Image) -> Image.Image:
    img = src.convert("RGBA")
    w, h = img.size
    pixels = img.load()
    seen = bytearray(w * h)
    q: deque[tuple[int, int]] = deque()

    def try_seed(x: int, y: int) -> None:
        idx = y * w + x
        if not seen[idx] and _is_background(*pixels[x, y]):
            seen[idx] = 1
            q.append((x, y))

    for x in range(w):
        try_seed(x, 0)
        try_seed(x, h - 1)
    for y in range(h):
        try_seed(0, y)
        try_seed(w - 1, y)

    while q:
        x, y = q.popleft()
        r, g, b, _ = pixels[x, y]
        pixels[x, y] = (r, g, b, 0)
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h:
                idx = ny * w + nx
                if not seen[idx] and _is_background(*pixels[nx, ny]):
                    seen[idx] = 1
                    q.append((nx, ny))

    return img.filter(ImageFilter.UnsharpMask(radius=1.4, percent=160, threshold=1))


def _save_png(img: Image.Image, path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.resize((size, size), Image.Resampling.LANCZOS).save(path, format="PNG", optimize=True)
    print(f"  [ok] {path} ({size}x{size})")


def generate(source: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source icon not found: {source}")

    print(f"Source: {source}")
    sharp = make_transparent(Image.open(source))
    app = _PROJECT_ROOT / "web" / "app"
    public = _PROJECT_ROOT / "web" / "public"

    _save_png(sharp, app / "icon.png", 1024)
    _save_png(sharp, app / "apple-icon.png", 180)
    _save_png(sharp, public / "icon-192.png", 192)
    _save_png(sharp, public / "icon-512.png", 512)

    ico_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128)]
    ico_imgs = [sharp.resize(s, Image.Resampling.LANCZOS) for s in ico_sizes]
    ico_path = app / "favicon.ico"
    ico_imgs[0].save(
        ico_path,
        format="ICO",
        sizes=ico_sizes,
        append_images=ico_imgs[1:],
    )
    print(f"  [ok] {ico_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate web icons with transparent background")
    parser.add_argument(
        "--source",
        default=str(_PROJECT_ROOT / "icon.png"),
        help="Source PNG (default: repo-root icon.png)",
    )
    args = parser.parse_args()
    generate(Path(args.source))


if __name__ == "__main__":
    main()
