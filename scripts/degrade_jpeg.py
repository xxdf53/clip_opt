"""
Batch image degradation for image folders.

The script reads images from an input folder, applies one or more degradation
operations, and writes outputs while preserving the original directory layout.

Examples:
    python scripts/degrade_jpeg.py --input ./test/airplane --output ./jpeg_test/airplane --ops jpeg --quality 75
    python scripts/degrade_jpeg.py --input ./test/airplane --output ./deg_test/airplane --ops jpeg blur noise --quality 75
"""

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
OPS = ("jpeg", "blur", "resize", "noise", "grayscale", "brightness", "contrast")


def parse_args():
    parser = argparse.ArgumentParser(description="Apply degradations to a folder of images.")
    parser.add_argument("--input", required=True, type=Path, help="Input image folder.")
    parser.add_argument("--output", required=True, type=Path, help="Output folder.")
    parser.add_argument("--ops", nargs="+", default=["jpeg"], choices=OPS, help="Degradation operations.")
    parser.add_argument("--quality", type=int, default=75, help="JPEG quality, 1-100.")
    parser.add_argument("--blur_radius", type=float, default=1.5, help="Gaussian blur radius.")
    parser.add_argument("--resize_scale", type=float, default=0.5, help="Downscale ratio before resizing back.")
    parser.add_argument("--noise_std", type=float, default=8.0, help="Gaussian noise std in pixel values.")
    parser.add_argument("--brightness", type=float, default=0.8, help="Brightness factor.")
    parser.add_argument("--contrast", type=float, default=0.8, help="Contrast factor.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for noise.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    return parser.parse_args()


def jpeg_degrade(image, quality):
    image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def resize_degrade(image, scale):
    width, height = image.size
    small_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    image = image.resize(small_size, Image.Resampling.BICUBIC)
    return image.resize((width, height), Image.Resampling.BICUBIC)


def noise_degrade(image, std, rng):
    array = np.asarray(image.convert("RGB")).astype(np.float32)
    array += rng.normal(0.0, std, array.shape).astype(np.float32)
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def apply_degradations(image, args, rng):
    image = image.convert("RGB")
    for op in args.ops:
        if op == "jpeg":
            image = jpeg_degrade(image, args.quality)
        elif op == "blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=args.blur_radius))
        elif op == "resize":
            image = resize_degrade(image, args.resize_scale)
        elif op == "noise":
            image = noise_degrade(image, args.noise_std, rng)
        elif op == "grayscale":
            image = image.convert("L").convert("RGB")
        elif op == "brightness":
            image = ImageEnhance.Brightness(image).enhance(args.brightness)
        elif op == "contrast":
            image = ImageEnhance.Contrast(image).enhance(args.contrast)
    return image


def iter_images(root):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def validate_args(args):
    if not args.input.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {args.input}")
    if not 1 <= args.quality <= 100:
        raise ValueError("--quality must be between 1 and 100")
    if not 0 < args.resize_scale <= 1:
        raise ValueError("--resize_scale must be in (0, 1]")
    if args.noise_std < 0:
        raise ValueError("--noise_std must be >= 0")


def main():
    args = parse_args()
    validate_args(args)

    rng = np.random.default_rng(args.seed)
    image_paths = list(iter_images(args.input))
    done, skipped, failed = 0, 0, 0

    for src in image_paths:
        dst = (args.output / src.relative_to(args.input)).with_suffix(".jpg")
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            degraded = apply_degradations(Image.open(src), args, rng)
            degraded.save(dst, format="JPEG", quality=args.quality)
            done += 1
        except Exception as exc:
            failed += 1
            print(f"[failed] {src}: {exc}")

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Ops: {' '.join(args.ops)}")
    print(f"Images: {len(image_paths)}, done: {done}, skipped: {skipped}, failed: {failed}")


if __name__ == "__main__":
    main()