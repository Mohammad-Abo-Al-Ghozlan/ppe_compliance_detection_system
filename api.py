"""FastAPI REST API for PPE Detection.

Provides endpoints for image/video inference, health checks, and metrics.
Includes Pydantic schemas and full OpenAPI documentation.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ultralytics import YOLO

from inference import detect_violations
from utils import (
    DEFAULT_CONFIG_PATH,
    PipelineConfig,
    get_device_info,
    load_config,
    setup_logger,
)

# Global state
app_state: dict[str, Any] = {
    "model": None,
    "config": None,
    "model_name": "",
    "start_time": 0.0,
    "device_info": {},
}


# =============================================================================
# Pydantic Schemas
# =============================================================================


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Top-left X coordinate", json_schema_extra={"examples": [100.0]})
    y1: float = Field(..., description="Top-left Y coordinate", json_schema_extra={"examples": [50.0]})
    x2: float = Field(..., description="Bottom-right X coordinate", json_schema_extra={"examples": [200.0]})
    y2: float = Field(..., description="Bottom-right Y coordinate", json_schema_extra={"examples": [150.0]})


class Detection(BaseModel):
    class_id: int = Field(..., description="Class index", json_schema_extra={"examples": [3]})
    class_name: str = Field(..., description="Class label", json_schema_extra={"examples": ["helmet"]})
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence", json_schema_extra={"examples": [0.92]})
    bbox: BoundingBox
    area: float = Field(..., description="Bounding box area in pixels", json_schema_extra={"examples": [10000.0]})


class PredictionResponse(BaseModel):
    success: bool = True
    predictions: list[Detection]
    detection_count: int
    violation_count: int = Field(..., description="Number of PPE violations detected")
    violations: list[str] = Field(default_factory=list, description="List of violation details")
    processing_time_ms: float
    image_size: dict[str, int]


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"examples": ["healthy"]})
    model_loaded: bool
    model_name: str
    device: str = Field(..., json_schema_extra={"examples": ["cuda:0"]})
    gpu_available: bool
    uptime_seconds: float


class MetricsResponse(BaseModel):
    model_name: str
    mAP50: float | None = None
    mAP50_95: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    inference_fps: float | None = None
    model_size_mb: float | None = None


class ClassesResponse(BaseModel):
    classes: dict[int, str]
    count: int


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: str | None = None


# =============================================================================
# Lifecycle and Setup
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger = logging.getLogger("api")
    
    # Extract config and model from app state (set before uvicorn.run)
    config = app_state.get("config")
    model_path = app_state.get("model_path")
    
    if not config or not model_path:
        logger.error("API started without proper configuration state.")
        yield
        return
        
    app_state["start_time"] = time.time()
    app_state["device_info"] = get_device_info()
    
    try:
        logger.info(f"Loading model from {model_path}...")
        model = YOLO(str(model_path))
        # Warmup
        model.predict(str(config.dataset.test_path / "images"), imgsz=config.dataset.img_size, verbose=False, max_det=1)
        
        app_state["model"] = model
        app_state["model_name"] = Path(model_path).parent.parent.name
        logger.info("Model loaded and warmed up successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        
    yield
    
    logger.info("Shutting down API...")


app = FastAPI(
    title="PPE Detection API",
    description="Enterprise API for Personal Protective Equipment object detection.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check and status."""
    device_info = app_state.get("device_info", {})
    return HealthResponse(
        status="healthy" if app_state.get("model") else "model_error",
        model_loaded=app_state.get("model") is not None,
        model_name=app_state.get("model_name", "unknown"),
        device=device_info.get("device", "cpu"),
        gpu_available=device_info.get("cuda_available", False),
        uptime_seconds=time.time() - app_state.get("start_time", time.time())
    )


@app.get("/classes", response_model=ClassesResponse)
async def get_classes():
    """Get list of supported classes."""
    config: PipelineConfig | None = app_state.get("config")
    if not config:
        raise HTTPException(status_code=500, detail="Config not loaded")
        
    classes = {i: name for i, name in enumerate(config.dataset.names)}
    return ClassesResponse(classes=classes, count=len(classes))


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    conf: float = Query(None, description="Confidence threshold", ge=0.0, le=1.0),
    iou: float = Query(None, description="NMS IoU threshold", ge=0.0, le=1.0)
):
    """Run inference on a single uploaded image."""
    model: YOLO | None = app_state.get("model")
    config: PipelineConfig | None = app_state.get("config")
    
    if not model or not config:
        raise HTTPException(status_code=503, detail="Model not initialized")
        
    if file.size is not None and file.size > config.api.max_file_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max {config.api.max_file_size_mb}MB")
        
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    filename = file.filename or "unknown.jpg"
    ext = Path(filename).suffix.lower()
    if ext not in valid_exts:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Supported: {valid_exts}")
        
    t0 = time.perf_counter()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name
        
    try:
        conf_thresh = conf if conf is not None else config.inference.conf
        iou_thresh = iou if iou is not None else config.inference.iou
        
        results = model.predict(temp_path, imgsz=config.dataset.img_size, conf=conf_thresh, iou=iou_thresh, verbose=False)
        result = results[0]
        
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class_id": int(box.cls[0]),
                    "class_name": config.dataset.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "area": (x2 - x1) * (y2 - y1)
                })
                
        # Detect violations
        v_count, v_list = detect_violations(detections)
        
        t_ms = (time.perf_counter() - t0) * 1000
        
        # We need actual image size
        img = cv2.imread(temp_path)
        h, w = img.shape[:2] if img is not None else (0, 0)
        
        return PredictionResponse(
            predictions=[Detection(**d) for d in detections],
            detection_count=len(detections),
            violation_count=v_count,
            violations=v_list,
            processing_time_ms=t_ms,
            image_size={"width": w, "height": h}
        )
    finally:
        Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start PPE Detection API.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model", type=str, required=True, help="Path to weights")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logger("api", config)
    
    # Set global state for FastAPI
    app_state["config"] = config
    app_state["model_path"] = Path(args.model)
    
    uvicorn.run("api:app", host=config.api.host, port=args.port, workers=config.api.workers)
