#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image Compression Tool for MAI-UI Dataset.

This script compresses images in a dataset directory to reduce storage and
improve data loading performance.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def compress_image(
    input_path: str,
    output_path: str,
    max_size: tuple[int, int] | None = None,
    quality: int = 85,
    format: str = "JPEG",
) -> tuple[int, int]:
    """Compress a single image.
    
    Args:
        input_path: Path to input image.
        output_path: Path to save compressed image.
        max_size: Maximum (width, height) for resizing.
        quality: Compression quality (0-100).
        format: Output format (JPEG or PNG).
        
    Returns:
        Tuple of (original_size, compressed_size) in bytes.
    """
    original_size = os.path.getsize(input_path)
    
    img = Image.open(input_path)
    
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    
    if max_size:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if format == "JPEG":
        img.save(output_path, format="JPEG", quality=quality, optimize=True)
    else:
        img.save(output_path, format="PNG", optimize=True)
    
    compressed_size = os.path.getsize(output_path)
    
    return original_size, compressed_size


def compress_directory(
    input_dir: str,
    output_dir: str,
    max_size: tuple[int, int] | None = None,
    quality: int = 85,
    format: str = "JPEG",
    skip_existing: bool = True,
) -> dict[str, int]:
    """Compress all images in a directory.
    
    Args:
        input_dir: Input directory containing images.
        output_dir: Output directory for compressed images.
        max_size: Maximum (width, height) for resizing.
        quality: Compression quality (0-100).
        format: Output format (JPEG or PNG).
        skip_existing: Skip files that already exist in output directory.
        
    Returns:
        Statistics dictionary.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.rglob(f"*{ext}"))
        image_files.extend(input_path.rglob(f"*{ext.upper()}"))
    
    total_original_size = 0
    total_compressed_size = 0
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for img_file in tqdm(image_files, desc="Compressing images"):
        try:
            rel_path = img_file.relative_to(input_path)
            
            if format == "JPEG":
                output_file = output_path / rel_path.with_suffix(".jpg")
            else:
                output_file = output_path / rel_path
            
            if skip_existing and output_file.exists():
                skipped_count += 1
                continue
            
            original_size, compressed_size = compress_image(
                str(img_file),
                str(output_file),
                max_size=max_size,
                quality=quality,
                format=format,
            )
            
            total_original_size += original_size
            total_compressed_size += compressed_size
            processed_count += 1
        
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            error_count += 1
    
    stats = {
        "total_files": len(image_files),
        "processed": processed_count,
        "skipped": skipped_count,
        "errors": error_count,
        "original_size_mb": total_original_size / (1024 * 1024),
        "compressed_size_mb": total_compressed_size / (1024 * 1024),
        "compression_ratio": (
            1 - (total_compressed_size / total_original_size)
        ) if total_original_size > 0 else 0,
    }
    
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress images in a dataset directory"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input directory containing images",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output directory for compressed images",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        help="Maximum width for resizing",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        help="Maximum height for resizing",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="JPEG quality (0-100, default: 85)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["JPEG", "PNG"],
        default="JPEG",
        help="Output format (default: JPEG)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of original directory",
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    if args.backup and input_dir != output_dir:
        backup_dir = input_dir.parent / f"{input_dir.name}_backup"
        print(f"Creating backup: {backup_dir}")
        if backup_dir.exists():
            print(f"Warning: Backup directory already exists")
        else:
            shutil.copytree(input_dir, backup_dir)
            print("Backup created successfully")
    
    max_size = None
    if args.max_width and args.max_height:
        max_size = (args.max_width, args.max_height)
    
    print(f"\nCompressing images:")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    if max_size:
        print(f"  Max size: {max_size[0]}x{max_size[1]}")
    print(f"  Quality: {args.quality}")
    print(f"  Format: {args.format}\n")
    
    stats = compress_directory(
        str(input_dir),
        str(output_dir),
        max_size=max_size,
        quality=args.quality,
        format=args.format,
        skip_existing=not args.overwrite,
    )
    
    print(f"\n{'='*60}")
    print("Compression Statistics")
    print(f"{'='*60}")
    print(f"Total files: {stats['total_files']}")
    print(f"Processed: {stats['processed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print(f"Original size: {stats['original_size_mb']:.2f} MB")
    print(f"Compressed size: {stats['compressed_size_mb']:.2f} MB")
    print(f"Compression ratio: {stats['compression_ratio']:.1%}")
    print(f"Space saved: {stats['original_size_mb'] - stats['compressed_size_mb']:.2f} MB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
