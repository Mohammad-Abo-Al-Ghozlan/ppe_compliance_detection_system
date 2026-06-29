"""Comprehensive model evaluation module for PPE Detection.

Computes mAP, Precision, Recall, FPS, size, and resource metrics.
Generates evaluation plots and error analysis.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from utils import (
    DEFAULT_CONFIG_PATH,
    PipelineConfig,
    ensure_dir,
    get_model_info,
    get_resource_usage,
    load_config,
    progress_bar,
    save_json,
    setup_logger,
)


def measure_inference_speed(
    weights_path: Path, test_images: list[Path], imgsz: int
) -> dict[str, float]:
    """Measure inference speed over test set.

    Args:
        weights_path: Path to model weights.
        test_images: List of test image paths.
        imgsz: Inference image size.

    Returns:
        Dict with fps and latency_ms.
    """
    if not test_images:
        return {"fps": 0.0, "latency_ms": 0.0}

    model = YOLO(str(weights_path))

    # Warmup
    warmup_imgs = test_images[: min(10, len(test_images))]
    for img in warmup_imgs:
        _ = model.predict(str(img), imgsz=imgsz, verbose=False)

    # Timed run
    start_time = time.perf_counter()
    for img in progress_bar(test_images, desc="Measuring speed"):
        _ = model.predict(str(img), imgsz=imgsz, verbose=False)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    latency_ms = (total_time / len(test_images)) * 1000
    fps = 1000 / latency_ms if latency_ms > 0 else 0

    return {"fps": float(fps), "latency_ms": float(latency_ms)}


def get_training_time(results_csv: Path) -> float:
    """Extract training time from Ultralytics results.csv if possible.

    This is an approximation as ultralytics doesn't save total time directly
    in the csv in an easily parseable format by default without parsing logs.
    We just return 0.0 here and let comparison script handle missing values,
    or read from tensorboard/logs in a more complex setup.
    """
    # For now, return 0.0
    return 0.0


def evaluate_model(
    weights_path: Path, config: PipelineConfig, model_name: str | None = None
) -> dict[str, Any]:
    """Evaluate a single model.

    Args:
        weights_path: Path to trained weights.
        config: Pipeline configuration.
        model_name: Optional explicit model name.

    Returns:
        Evaluation metrics dictionary.
    """
    logger = setup_logger("evaluate_models", config)
    logger.info(f"Evaluating {weights_path}...")

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    if model_name is None:
        model_name = weights_path.parent.parent.name

    out_dir = config.get_outputs_dir() / "reports"
    ensure_dir(out_dir)

    # Get model info
    info = get_model_info(model_name, weights_path)

    # Standard Ultralytics val
    model = YOLO(str(weights_path))

    logger.info("Running validation on test split...")
    results = model.val(
        data=str(config.dataset.data_yaml),
        split="test",
        imgsz=config.dataset.img_size,
        conf=config.evaluation.conf_threshold,
        iou=config.evaluation.iou_threshold,
        max_det=config.evaluation.max_det,
        verbose=False,
        save_json=True,
    )

    # Measure speed
    test_dir = config.dataset.test_path
    from utils import get_images_in_dir

    test_images = get_images_in_dir(test_dir)
    speed_metrics = measure_inference_speed(weights_path, test_images, config.dataset.img_size)

    # Resources
    resources = get_resource_usage()

    # Training time
    train_dir = weights_path.parent.parent
    train_time = get_training_time(train_dir / "results.csv")

    metrics: dict[str, Any] = {
        "model_name": model_name,
        "weights": str(weights_path),
        "size_mb": info.file_size_mb,
        "params": info.parameter_count,
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "f1": float(
            2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr + 1e-10)
        ),
        "fps": speed_metrics["fps"],
        "latency_ms": speed_metrics["latency_ms"],
        "training_time_hrs": train_time,
        "resources": resources,
        "per_class": {},
    }

    for i, class_name in enumerate(config.dataset.names):
        if i < len(results.box.ap):
            metrics["per_class"][class_name] = {
                "ap50": float(results.box.ap50[i]) if len(results.box.ap50) > i else 0.0,
                "ap50_95": float(results.box.ap[i]) if len(results.box.ap) > i else 0.0,
            }

    # Save results
    report_path = out_dir / f"evaluation_{model_name}.json"
    save_json(metrics, report_path)
    logger.info(f"Saved evaluation report to {report_path}")

    # Error analysis and visualizations would follow here
    # (Generating confusion matrix, PR curves, etc. usually done via ultralytics built-ins
    # which are saved to the runs/val directory automatically)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to config file"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model weights to evaluate. If omitted, evaluates all in models dir.",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        logger = setup_logger("evaluate_models", config)

        if args.model:
            evaluate_model(Path(args.model), config)
        else:
            models_dir = config.get_models_dir()
            if not models_dir.exists():
                logger.error(f"Models directory not found: {models_dir}")
                sys.exit(1)

            # Find all best.pt files
            weights_files = list(models_dir.glob("*/weights/best.pt"))
            if not weights_files:
                logger.error(f"No trained models found in {models_dir}")
                sys.exit(1)

            for weights_path in weights_files:
                evaluate_model(weights_path, config)

        sys.exit(0)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
