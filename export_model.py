"""Model export module for PPE Detection.

Exports PyTorch weights to ONNX and TorchScript formats.
Generates an export report with model size and inference compatibility.
"""

from __future__ import annotations

import argparse
import datetime
import shutil
import sys
from pathlib import Path
from typing import Any, cast

from utils import (
    DEFAULT_CONFIG_PATH,
    DetectorRegistry,
    ModelConfig,
    PipelineConfig,
    UltralyticsDetector,
    ensure_dir,
    load_config,
    save_json,
    setup_logger,
)


def export_model(
    config: PipelineConfig, weights_path: Path, formats: list[str] | None = None
) -> dict[str, Any]:
    """Export model to deployment formats.

    Args:
        config: Pipeline configuration.
        weights_path: Path to PyTorch model weights.
        formats: List of formats to export to (overrides config).

    Returns:
        Export report dictionary.
    """
    logger = setup_logger("export_model", config)
    logger.info(f"Starting export for {weights_path}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model_name = weights_path.parent.parent.name
    if model_name == "weights":  # In case someone passes just the file in root
        model_name = weights_path.stem

    if formats:
        config.export.formats = formats

    # Use UltralyticsDetector for export
    model_cfg = ModelConfig(name=model_name, weights=str(weights_path))

    try:
        detector_class = DetectorRegistry.get(model_name)
    except KeyError:
        detector_class = UltralyticsDetector

    detector = detector_class(model_cfg)

    # Run export
    report = detector.export(weights_path, config)

    # Add metadata
    report["model_name"] = model_name
    report["source_weights"] = str(weights_path)
    report["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Copy exported files to centralized outputs/exports dir
    out_dir = config.get_outputs_dir() / "exports"
    ensure_dir(out_dir)

    # Copy PyTorch weights
    pt_dest = out_dir / f"{model_name}.pt"
    shutil.copy2(weights_path, pt_dest)

    # Update report paths
    for exp in report["exports"]:
        if exp["format"] == "pytorch":
            exp["file_path"] = str(pt_dest)
        elif exp["success"] and exp["file_path"]:
            src_file = Path(exp["file_path"])
            if src_file.exists():
                dest_file = out_dir / f"{model_name}{src_file.suffix}"
                shutil.copy2(src_file, dest_file)
                exp["file_path"] = str(dest_file)
                logger.info(f"Copied {src_file.name} to {dest_file}")

    # Save report
    report_path = out_dir / "export_report.json"
    save_json(report, report_path)
    logger.info(f"Saved export report to {report_path}")

    return cast(dict[str, Any], report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PPE Detection models.")
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to config file"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument(
        "--formats", type=str, nargs="+", help="Formats to export (e.g., onnx torchscript)"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        export_model(config, Path(args.model), args.formats)
        sys.exit(0)
    except Exception as e:
        print(f"Error during export: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
