"""Multi-source inference engine for PPE Detection.

Supports single images, folders, videos, webcam, RTSP, and YouTube.
Includes violation detection logic (e.g., person without helmet).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yt_dlp
from ultralytics import YOLO

from utils import (
    DEFAULT_CONFIG_PATH,
    PipelineConfig,
    ensure_dir,
    load_config,
    progress_bar,
    save_json,
    setup_logger,
)


def extract_youtube_url(url: str) -> str | None:
    """Extract direct video URL from YouTube link."""
    ydl_opts = {
        "format": "best[ext=mp4]",
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info and "url" in info:
                return str(info["url"])
    except Exception as e:
        print(f"Failed to extract YouTube URL: {e}")
    return None


def calculate_iou(box1: dict[str, float], box2: dict[str, float]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])

    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0

    return intersection / union


def detect_violations(detections: list[dict[str, Any]]) -> tuple[int, list[str]]:
    """Detect PPE violations (person without required PPE).

    Required PPE (must overlap with person bbox): helmet, vest

    Returns:
        Tuple of (violation_count, list of violation descriptions).
    """
    violations = []

    persons = [d for d in detections if d["class_name"] == "person"]
    ppe_items = [
        d
        for d in detections
        if d["class_name"] in ["helmet", "vest", "goggles", "boots", "gloves"]
    ]

    for i, person in enumerate(persons):
        has_helmet = False
        has_vest = False

        for ppe in ppe_items:
            # Overlap threshold
            if calculate_iou(person["bbox"], ppe["bbox"]) > 0.1:
                if ppe["class_name"] == "helmet":
                    has_helmet = True
                elif ppe["class_name"] == "vest":
                    has_vest = True

        missing = []
        if not has_helmet:
            missing.append("helmet")
        if not has_vest:
            missing.append("vest")

        if missing:
            violations.append(f"Person {i+1} missing: {', '.join(missing)}")

    return len(violations), violations


def annotate_frame(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    violations: list[str],
    fps: float | None = None,
) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    annotated = frame.copy()

    # Class colors (BGR for cv2)
    colors = {
        "person": (233, 69, 96),  # e94560
        "helmet": (15, 52, 96),  # 0f3460
        "vest": (22, 33, 62),  # 16213e
        "goggles": (83, 52, 131),  # 533483
        "boots": (216, 180, 0),  # 00b4d8
        "gloves": (0, 180, 216),
    }
    default_color = (0, 255, 0)

    for det in detections:
        box = det["bbox"]
        name = det["class_name"]
        conf = det["confidence"]

        color = colors.get(name, default_color)

        x1, y1 = int(box["x1"]), int(box["y1"])
        x2, y2 = int(box["x2"]), int(box["y2"])

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label = f"{name} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w, y1), color, -1)

        # Draw label text
        cv2.putText(
            annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    # Draw FPS
    if fps is not None:
        cv2.putText(
            annotated, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

    # Draw violations
    if violations:
        y_offset = 60
        cv2.putText(
            annotated,
            f"VIOLATIONS: {len(violations)}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        for v in violations:
            y_offset += 25
            cv2.putText(
                annotated, v, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

    return annotated


def run_inference(
    config: PipelineConfig,
    model_path: Path,
    source: str | list[str],
    conf: float | None = None,
    iou: float | None = None,
    save: bool = False,
    save_video: bool = False,
    show: bool = False,
) -> None:
    """Run multi-source inference."""
    logger = setup_logger("inference", config)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))

    out_dir = config.get_outputs_dir() / "predictions"
    ensure_dir(out_dir)
    img_out_dir = out_dir / "images"
    ensure_dir(img_out_dir)

    conf_thresh = conf if conf is not None else config.inference.conf
    iou_thresh = iou if iou is not None else config.inference.iou

    # Handle multiple images vs single source
    if isinstance(source, list):
        if len(source) == 1:
            source_str = source[0]
        else:
            # Treat as list of images
            all_results = []
            for img_path in source:
                res = _process_image(
                    model,
                    img_path,
                    conf_thresh,
                    iou_thresh,
                    config,
                    save,
                    show,
                    img_out_dir,
                    logger,
                )
                all_results.append(res)
            _save_reports(all_results, out_dir, logger)
            return
    else:
        source_str = str(source)

    # Detect source type
    if (
        source_str.isdigit()
        or source_str.startswith("rtsp://")
        or ".mp4" in source_str.lower()
        or "youtube" in source_str.lower()
        or "youtu.be" in source_str.lower()
    ):
        _process_video_stream(
            model, source_str, conf_thresh, iou_thresh, config, save_video, show, out_dir, logger
        )
    else:
        path = Path(source_str)
        if path.is_dir():
            all_results = []
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_path in progress_bar(list(path.glob(ext)), desc="Processing folder"):
                    res = _process_image(
                        model,
                        str(img_path),
                        conf_thresh,
                        iou_thresh,
                        config,
                        save,
                        False,
                        img_out_dir,
                        logger,
                    )
                    all_results.append(res)
            _save_reports(all_results, out_dir, logger)
        else:
            res = _process_image(
                model, source_str, conf_thresh, iou_thresh, config, save, show, img_out_dir, logger
            )
            _save_reports([res], out_dir, logger)


def _process_image(
    model: YOLO,
    img_path: str,
    conf: float,
    iou: float,
    config: PipelineConfig,
    save: bool,
    show: bool,
    out_dir: Path,
    logger,
) -> dict:
    """Process single image."""
    img = cv2.imread(img_path)
    if img is None:
        logger.error(f"Failed to read image: {img_path}")
        return {}

    results = model.predict(img, imgsz=config.dataset.img_size, conf=conf, iou=iou, verbose=False)
    result = results[0]

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "class_id": int(box.cls[0]),
                    "class_name": config.dataset.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )

    v_count, v_list = detect_violations(detections)

    annotated = annotate_frame(img, detections, v_list)

    if save:
        out_path = out_dir / f"pred_{Path(img_path).name}"
        cv2.imwrite(str(out_path), annotated)

    if show:
        cv2.imshow("Prediction", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "source": img_path,
        "detections": detections,
        "violation_count": v_count,
        "violations": v_list,
    }


def _process_video_stream(
    model: YOLO,
    source_str: str,
    conf: float,
    iou: float,
    config: PipelineConfig,
    save_video: bool,
    show: bool,
    out_dir: Path,
    logger,
) -> None:
    """Process video or stream."""
    stream_src: str | int | None = source_str
    if "youtube" in source_str.lower() or "youtu.be" in source_str.lower():
        logger.info("Extracting YouTube URL...")
        stream_src = extract_youtube_url(source_str)
        if not stream_src:
            logger.error("Failed to extract YouTube video.")
            return

    if source_str.isdigit():
        stream_src = int(source_str)

    cap = cv2.VideoCapture(stream_src)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source_str}")
        return

    writer = None
    if save_video:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        out_path = out_dir / "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        logger.info(f"Saving video to {out_path}")

    frames_processed = 0
    t0 = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                frame, imgsz=config.dataset.img_size, conf=conf, iou=iou, verbose=False
            )
            result = results[0]

            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(
                        {
                            "class_id": int(box.cls[0]),
                            "class_name": config.dataset.names[int(box.cls[0])],
                            "confidence": float(box.conf[0]),
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        }
                    )

            _, v_list = detect_violations(detections)

            frames_processed += 1
            current_fps = frames_processed / (time.perf_counter() - t0)

            annotated = annotate_frame(frame, detections, v_list, fps=current_fps)

            if writer:
                writer.write(annotated)

            if show:
                cv2.imshow("PPE Detection Inference", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()


def _save_reports(results: list[dict], out_dir: Path, logger) -> None:
    """Save inference results to JSON and CSV."""
    if not results:
        return

    save_json(results, out_dir / "results.json")

    # Flatten for CSV
    flat = []
    for res in results:
        for d in res.get("detections", []):
            flat.append(
                {
                    "source": res["source"],
                    "class_id": d["class_id"],
                    "class_name": d["class_name"],
                    "confidence": d["confidence"],
                    "x1": d["bbox"]["x1"],
                    "y1": d["bbox"]["y1"],
                    "x2": d["bbox"]["x2"],
                    "y2": d["bbox"]["y2"],
                    "violations": len(res.get("violations", [])),
                }
            )

    if flat:
        df = pd.DataFrame(flat)
        df.to_csv(out_dir / "results.csv", index=False)
        logger.info(f"Saved predictions to {out_dir}/results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PPE Detection inference.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model", type=str, required=True, help="Path to weights")
    parser.add_argument("--source", type=str, nargs="+", required=True, help="Input source(s)")
    parser.add_argument("--conf", type=float, help="Confidence threshold")
    parser.add_argument("--iou", type=float, help="NMS IoU threshold")
    parser.add_argument("--save", action="store_true", help="Save annotated images")
    parser.add_argument("--save-video", action="store_true", help="Save annotated video")
    parser.add_argument("--show", action="store_true", help="Display results")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        run_inference(
            config=config,
            model_path=Path(args.model),
            source=args.source,
            conf=args.conf,
            iou=args.iou,
            save=args.save,
            save_video=args.save_video,
            show=args.show,
        )
        sys.exit(0)
    except Exception as e:
        print(f"Inference error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
