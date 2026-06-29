"""Data cleaning and validation module for PPE Detection.

Provides strict validation of the YOLO dataset (segmentation format).
Aborts if critical inconsistencies are found.
Generates statistics and visualizations of the dataset.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

from utils import (
    PipelineConfig,
    compute_file_hash,
    ensure_dir,
    get_images_in_dir,
    get_label_path_for_image,
    load_config,
    progress_bar,
    save_json,
    setup_logger,
)


def validate_dataset(config: PipelineConfig) -> bool:
    """Validate dataset and generate report/visualizations.

    Args:
        config: Pipeline configuration.

    Returns:
        True if validation passes, False otherwise.
    """
    logger = setup_logger("data_cleaning", config)
    logger.info("Starting dataset validation...")

    nc = config.dataset.nc
    names = config.dataset.names
    img_size = config.dataset.img_size

    errors: list[str] = []
    warnings: list[str] = []

    splits = {
        "train": config.dataset.train_path,
        "val": config.dataset.val_path,
        "test": config.dataset.test_path,
    }

    report: dict[str, Any] = {
        "splits": {},
        "class_distribution": {name: 0 for name in names},
        "objects_per_image": [],
        "bbox_areas": [],
        "resolutions": set(),
        "duplicates": [],
    }

    image_hashes: dict[str, list[str]] = defaultdict(list)
    unique_class_ids: set[int] = set()
    heatmap_grid = np.zeros((img_size, img_size), dtype=np.float32)

    for split_name, split_path in splits.items():
        if not split_path.exists():
            errors.append(f"Split path not found: {split_path}")
            continue

        images = get_images_in_dir(split_path)
        labels_dir = split_path.parent / "labels"
        
        if labels_dir.exists():
            label_files = {f.name for f in labels_dir.iterdir() if f.is_file() and f.suffix == ".txt"}
        else:
            label_files = set()
            errors.append(f"Labels directory not found for split {split_name}: {labels_dir}")

        split_stats = {
            "images": len(images),
            "labels": len(label_files),
        }
        report["splits"][split_name] = split_stats
        
        logger.info(f"Processing {split_name} split ({len(images)} images)...")

        for img_path in progress_bar(images, desc=f"{split_name} images"):
            # Corrupted image check and resolution
            try:
                with Image.open(img_path) as img:
                    img.verify()
                
                # We need to open it again to get size as verify() doesn't always load full header
                with Image.open(img_path) as img:
                    w, h = img.size
                    report["resolutions"].add(f"{w}x{h}")
                    if w != img_size or h != img_size:
                        warnings.append(f"Image {img_path.name} is {w}x{h}, expected {img_size}x{img_size}")
            except Exception as e:
                errors.append(f"Corrupted image {img_path.name}: {e}")
                continue
                
            # Duplicate detection
            img_hash = compute_file_hash(img_path)
            image_hashes[img_hash].append(str(img_path.name))

            # Label matching
            label_path = get_label_path_for_image(img_path)
            if not label_path.exists():
                errors.append(f"Missing label file for image: {img_path.name}")
                continue
                
            label_files.discard(label_path.name)

            # Check label contents (YOLO segmentation format)
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception as e:
                errors.append(f"Cannot read label {label_path.name}: {e}")
                continue

            if not lines:
                warnings.append(f"Empty label file: {label_path.name}")
                report["objects_per_image"].append(0)
                continue

            report["objects_per_image"].append(len(lines))

            for line_idx, line in enumerate(lines):
                parts = line.strip().split()
                if not parts:
                    continue
                
                try:
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    errors.append(f"Invalid format in {label_path.name} line {line_idx+1}")
                    continue

                if class_id < 0 or class_id >= nc:
                    errors.append(f"Invalid class ID {class_id} in {label_path.name}")
                    continue
                
                unique_class_ids.add(class_id)
                report["class_distribution"][names[class_id]] += 1

                if any(c < 0.0 or c > 1.0 for c in coords):
                    errors.append(f"Coordinates outside [0, 1] in {label_path.name} line {line_idx+1}")
                
                # Bbox area from polygon
                if len(coords) >= 4:
                    xs = coords[0::2]
                    ys = coords[1::2]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    area = (max_x - min_x) * (max_y - min_y)
                    report["bbox_areas"].append(area)
                    
                    # Update heatmap
                    cx = int((min_x + max_x) / 2 * (img_size - 1))
                    cy = int((min_y + max_y) / 2 * (img_size - 1))
                    heatmap_grid[cy, cx] += 1

        for orphan_label in label_files:
            errors.append(f"Label without image in {split_name}: {orphan_label}")

    # Process duplicates
    for img_hash, paths in image_hashes.items():
        if len(paths) > 1:
            report["duplicates"].append(paths)
            warnings.append(f"Found {len(paths)} duplicate images")

    # Check class consistency
    if len(unique_class_ids) != nc:
        errors.append(f"Dataset claims {nc} classes, but only found {len(unique_class_ids)} unique classes in labels.")

    # Convert sets for JSON
    report["resolutions"] = list(report["resolutions"])
    
    # Calculate stats
    objs = report["objects_per_image"]
    if objs:
        report["objects_stats"] = {
            "min": int(np.min(objs)),
            "max": int(np.max(objs)),
            "mean": float(np.mean(objs)),
            "std": float(np.std(objs)),
        }
    
    areas = report["bbox_areas"]
    if areas:
        report["area_stats"] = {
            "min": float(np.min(areas)),
            "max": float(np.max(areas)),
            "mean": float(np.mean(areas)),
            "std": float(np.std(areas)),
        }

    # Save report
    out_dir = config.get_outputs_dir() / "reports"
    ensure_dir(out_dir)
    save_json(report, out_dir / "dataset_report.json")
    
    # Visualizations
    viz_dir = config.get_visualizations_dir()
    ensure_dir(viz_dir)
    
    _generate_visualizations(report, heatmap_grid, viz_dir, names)

    # Log results
    for warn_msg in warnings[:10]:
        logger.warning(warn_msg)
    if len(warnings) > 10:
        logger.warning(f"... and {len(warnings) - 10} more warnings.")

    if errors:
        for err_msg in errors[:10]:
            logger.error(err_msg)
        if len(errors) > 10:
            logger.error(f"... and {len(errors) - 10} more errors.")
        logger.error("Dataset validation FAILED.")
        return False
    
    logger.info("Dataset validation PASSED.")
    return True


def _save_plot(fig: plt.Figure, name: str, viz_dir: Path) -> None:
    """Save figure in PNG and SVG formats."""
    fig.savefig(viz_dir / f"{name}.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(viz_dir / f"{name}.svg", format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _generate_visualizations(
    report: dict[str, Any], 
    heatmap_grid: np.ndarray, 
    viz_dir: Path, 
    names: list[str]
) -> None:
    """Generate dataset visualizations."""
    plt.style.use('dark_background')
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#1a1a2e", "figure.facecolor": "#1a1a2e", "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
    palette = ['#e94560', '#0f3460', '#16213e', '#533483', '#e94560', '#00b4d8']

    # Class distribution
    fig, ax = plt.subplots(figsize=(12, 8))
    classes = list(report["class_distribution"].keys())
    counts = list(report["class_distribution"].values())
    sns.barplot(y=classes, x=counts, hue=classes, palette=palette[:len(classes)] if len(classes)<=len(palette) else "viridis", legend=False, ax=ax)
    ax.set_title("Class Distribution", fontsize=16)
    ax.set_xlabel("Count")
    for i, v in enumerate(counts):
        ax.text(v + max(counts)*0.01, i, str(v), color='white', va='center')
    _save_plot(fig, "class_distribution", viz_dir)

    # Bbox areas
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(report["bbox_areas"], bins=50, color='#e94560', ax=ax, log_scale=True)
    ax.set_title("Bounding Box Area Distribution (Relative)", fontsize=16)
    ax.set_xlabel("Relative Area (log scale)")
    ax.set_ylabel("Frequency")
    _save_plot(fig, "bbox_size_histogram", viz_dir)

    # Objects per image
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(report["objects_per_image"], bins=range(0, max(report["objects_per_image"])+2) if report["objects_per_image"] else 10, color='#00b4d8', ax=ax, discrete=True)
    ax.set_title("Objects per Image Distribution", fontsize=16)
    ax.set_xlabel("Number of Objects")
    ax.set_ylabel("Frequency")
    _save_plot(fig, "objects_per_image_histogram", viz_dir)

    # Heatmap
    if np.sum(heatmap_grid) > 0:
        fig, ax = plt.subplots(figsize=(12, 12))
        heatmap_smoothed = cv2.GaussianBlur(heatmap_grid, (51, 51), 0)
        sns.heatmap(heatmap_smoothed, cmap="magma", xticklabels=False, yticklabels=False, ax=ax, cbar=True)
        ax.set_title("Spatial Distribution of Objects", fontsize=16)
        _save_plot(fig, "spatial_heatmap", viz_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate PPE Detection dataset.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        success = validate_dataset(config)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error during validation: {e}")
        sys.exit(1)
