"""Shared utilities and plugin architecture for the PPE Detection pipeline.

This module provides:
    - Configuration loading and validation (dataclass-based)
    - DetectorRegistry: plugin architecture for adding new detectors
    - DetectorProtocol: interface all detectors must implement
    - UltralyticsDetector: adapter for YOLO11 and RT-DETR models
    - Device detection, seed setting, logging, timing utilities
    - Model info extraction, path management, progress bar wrappers

All paths use pathlib.Path. No os.path usage. No print statements.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Protocol, runtime_checkable

import numpy as np
import psutil
import torch
import yaml
from tqdm import tqdm

# =============================================================================
# Constants
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_SEED = 42

# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class DatasetConfig:
    """Dataset configuration parsed from data.yaml (source of truth)."""

    data_yaml: Path
    train_path: Path
    val_path: Path
    test_path: Path
    nc: int
    names: list[str]
    img_size: int = 640

    def validate(self) -> list[str]:
        """Validate dataset configuration. Returns list of errors."""
        errors: list[str] = []
        if not self.data_yaml.exists():
            errors.append(f"data.yaml not found: {self.data_yaml}")
        if not self.train_path.exists():
            errors.append(f"Train path not found: {self.train_path}")
        if not self.val_path.exists():
            errors.append(f"Validation path not found: {self.val_path}")
        if not self.test_path.exists():
            errors.append(f"Test path not found: {self.test_path}")
        if self.nc != len(self.names):
            errors.append(
                f"Class count mismatch: nc={self.nc}, but {len(self.names)} names provided"
            )
        if self.nc <= 0:
            errors.append(f"Invalid class count: {self.nc}")
        return errors


@dataclass
class ModelConfig:
    """Single model configuration."""

    name: str
    weights: str


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 100
    batch_size: int = 16
    patience: int = 20
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    weight_decay: float = 0.0005
    momentum: float = 0.937
    amp: bool = True
    seed: int = 42
    workers: int = 8
    resume: bool = False


@dataclass
class AugmentationConfig:
    """Complementary augmentation settings.

    NOTE: hsv_v is 0.0 because brightness augmentation was already applied
    to the dataset during export from Roboflow.
    """

    mosaic: float = 1.0
    mixup: float = 0.15
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    perspective: float = 0.0
    erasing: float = 0.4


@dataclass
class HPOConfig:
    """Hyperparameter optimization configuration."""

    enabled: bool = False
    n_trials: int = 10
    metric: str = "metrics/mAP50-95"
    direction: str = "maximize"


@dataclass
class EvaluationConfig:
    """Evaluation thresholds."""

    conf_threshold: float = 0.25
    iou_threshold: float = 0.7
    max_det: int = 300


@dataclass
class InferenceConfig:
    """Inference defaults."""

    conf: float = 0.5
    iou: float = 0.45
    max_det: int = 300
    line_width: int = 2
    save_json: bool = True
    save_csv: bool = True


@dataclass
class ExportConfig:
    """Model export configuration."""

    formats: list[str] = field(default_factory=lambda: ["torchscript", "onnx"])
    dynamic: bool = False
    simplify: bool = True
    opset: int = 17
    half: bool = False


@dataclass
class APIConfig:
    """FastAPI settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_file_size_mb: int = 50


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    log_dir: str = "logs"


@dataclass
class PathsConfig:
    """Output path configuration."""

    models_dir: str = "models"
    outputs_dir: str = "outputs"
    visualizations_dir: str = "visualizations"


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    dataset: DatasetConfig
    models: list[ModelConfig]
    training: TrainingConfig
    augmentation: AugmentationConfig
    hpo: HPOConfig
    evaluation: EvaluationConfig
    inference: InferenceConfig
    export: ExportConfig
    api: APIConfig
    logging_config: LoggingConfig
    paths: PathsConfig
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)

    def get_models_dir(self) -> Path:
        """Get absolute path to models directory."""
        return self.project_root / self.paths.models_dir

    def get_outputs_dir(self) -> Path:
        """Get absolute path to outputs directory."""
        return self.project_root / self.paths.outputs_dir

    def get_visualizations_dir(self) -> Path:
        """Get absolute path to visualizations directory."""
        return self.project_root / self.paths.visualizations_dir

    def get_log_dir(self) -> Path:
        """Get absolute path to log directory."""
        return self.project_root / self.logging_config.log_dir


# =============================================================================
# Configuration Loading
# =============================================================================


def _parse_data_yaml(data_yaml_path: Path) -> dict[str, Any]:
    """Parse the dataset's data.yaml file (source of truth).

    Args:
        data_yaml_path: Absolute path to data.yaml.

    Returns:
        Parsed YAML content as dictionary.

    Raises:
        FileNotFoundError: If data.yaml does not exist.
        ValueError: If data.yaml is malformed.
    """
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"data.yaml is malformed: expected dict, got {type(data)}")

    required_keys = ["nc", "names"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"data.yaml missing required key: '{key}'")

    return data


def _resolve_dataset_path(data_yaml_dir: Path, relative_path: str) -> Path:
    """Resolve a dataset path relative to data.yaml's directory.

    Args:
        data_yaml_dir: Directory containing data.yaml.
        relative_path: Relative path string from data.yaml.

    Returns:
        Resolved absolute path.
    """
    resolved = (data_yaml_dir / relative_path).resolve()
    return resolved


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> PipelineConfig:
    """Load and validate the pipeline configuration.

    Reads config.yaml for pipeline settings and data.yaml for dataset-specific
    values. All dataset configuration comes from data.yaml as the source of truth.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Fully validated PipelineConfig instance.

    Raises:
        FileNotFoundError: If config files don't exist.
        ValueError: If configuration is invalid.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"config.yaml is malformed: expected dict, got {type(raw)}")

    project_root = config_path.parent.parent

    # Parse data.yaml (source of truth for dataset)
    data_yaml_rel = raw.get("dataset", {}).get("data_yaml", "archive/data.yaml")
    data_yaml_path = (project_root / data_yaml_rel).resolve()
    data_yaml = _parse_data_yaml(data_yaml_path)
    data_yaml_dir = data_yaml_path.parent

    # Resolve dataset paths from data.yaml
    train_path = _resolve_dataset_path(
        data_yaml_dir, data_yaml.get("train", "../train/images")
    )
    val_path = _resolve_dataset_path(
        data_yaml_dir, data_yaml.get("val", "../valid/images")
    )
    test_path = _resolve_dataset_path(
        data_yaml_dir, data_yaml.get("test", "../test/images")
    )

    dataset_cfg = DatasetConfig(
        data_yaml=data_yaml_path,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        nc=data_yaml["nc"],
        names=data_yaml["names"],
        img_size=raw.get("dataset", {}).get("img_size", 640),
    )

    # Parse model configs
    models_raw = raw.get("models", [])
    models = [ModelConfig(name=m["name"], weights=m["weights"]) for m in models_raw]

    # Parse all sub-configs with defaults
    training_raw = raw.get("training", {})
    training = TrainingConfig(**{k: v for k, v in training_raw.items()})

    aug_raw = raw.get("augmentation", {})
    augmentation = AugmentationConfig(**{k: v for k, v in aug_raw.items()})

    hpo_raw = raw.get("hpo", {})
    hpo = HPOConfig(**{k: v for k, v in hpo_raw.items()})

    eval_raw = raw.get("evaluation", {})
    evaluation = EvaluationConfig(**{k: v for k, v in eval_raw.items()})

    inf_raw = raw.get("inference", {})
    inference_cfg = InferenceConfig(**{k: v for k, v in inf_raw.items()})

    export_raw = raw.get("export", {})
    export_cfg = ExportConfig(**{k: v for k, v in export_raw.items()})

    api_raw = raw.get("api", {})
    api_cfg = APIConfig(**{k: v for k, v in api_raw.items()})

    log_raw = raw.get("logging", {})
    log_cfg = LoggingConfig(**{k: v for k, v in log_raw.items()})

    paths_raw = raw.get("paths", {})
    paths_cfg = PathsConfig(**{k: v for k, v in paths_raw.items()})

    config = PipelineConfig(
        dataset=dataset_cfg,
        models=models,
        training=training,
        augmentation=augmentation,
        hpo=hpo,
        evaluation=evaluation,
        inference=inference_cfg,
        export=export_cfg,
        api=api_cfg,
        logging_config=log_cfg,
        paths=paths_cfg,
        project_root=project_root,
    )

    return config


# =============================================================================
# Logging
# =============================================================================


def setup_logger(
    name: str,
    config: PipelineConfig | None = None,
    level: str | None = None,
) -> logging.Logger:
    """Set up a logger with file and console handlers.

    Args:
        name: Logger name (typically module __name__).
        config: Pipeline configuration for log directory and level.
        level: Override log level (e.g., "DEBUG", "INFO").

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_level = getattr(
        logging,
        level or (config.logging_config.level if config else "INFO"),
    )
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if config available)
    if config:
        log_dir = config.get_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"{name}.log", encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# Device Detection
# =============================================================================


def get_device() -> torch.device:
    """Auto-detect the best available compute device.

    Returns:
        torch.device: cuda if GPU available, mps for Apple Silicon, else cpu.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info() -> dict[str, Any]:
    """Get detailed device information.

    Returns:
        Dictionary with device name, type, GPU memory, CUDA version, etc.
    """
    device = get_device()
    info: dict[str, Any] = {
        "device": str(device),
        "device_type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / (1024**3), 2
        )
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()

    return info


# =============================================================================
# Reproducibility
# =============================================================================


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds for reproducibility across all frameworks.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# Timer
# =============================================================================


@dataclass
class TimerResult:
    """Result from a timed operation."""

    elapsed_seconds: float
    operation: str

    @property
    def elapsed_formatted(self) -> str:
        """Format elapsed time as human-readable string."""
        if self.elapsed_seconds < 60:
            return f"{self.elapsed_seconds:.2f}s"
        minutes = int(self.elapsed_seconds // 60)
        seconds = self.elapsed_seconds % 60
        if minutes < 60:
            return f"{minutes}m {seconds:.1f}s"
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h {minutes}m {seconds:.0f}s"


@contextmanager
def timer(operation: str = "Operation") -> Generator[TimerResult, None, None]:
    """Context manager for timing operations.

    Args:
        operation: Description of the operation being timed.

    Yields:
        TimerResult that is populated when the context exits.

    Example:
        with timer("Training") as t:
            train_model()
        print(t.elapsed_formatted)
    """
    result = TimerResult(elapsed_seconds=0.0, operation=operation)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_seconds = time.perf_counter() - start


# =============================================================================
# Model Info
# =============================================================================


@dataclass
class ModelInfo:
    """Information about a trained model."""

    name: str
    weights_path: Path
    file_size_mb: float
    parameter_count: int
    architecture: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "name": self.name,
            "weights_path": str(self.weights_path),
            "file_size_mb": self.file_size_mb,
            "parameter_count": self.parameter_count,
            "architecture": self.architecture,
        }


def get_model_info(name: str, weights_path: Path) -> ModelInfo:
    """Extract model information from weights file.

    Args:
        name: Model name identifier.
        weights_path: Path to model weights (.pt file).

    Returns:
        ModelInfo with size, parameters, and architecture details.
    """
    from ultralytics import YOLO

    file_size_mb = weights_path.stat().st_size / (1024 * 1024)
    model = YOLO(str(weights_path))

    param_count = sum(p.numel() for p in model.model.parameters())
    architecture = model.model.__class__.__name__ if hasattr(model, "model") else "unknown"

    return ModelInfo(
        name=name,
        weights_path=weights_path,
        file_size_mb=round(file_size_mb, 2),
        parameter_count=param_count,
        architecture=architecture,
    )


# =============================================================================
# Path Utilities
# =============================================================================


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist.

    Args:
        path: Directory path to create.

    Returns:
        The same path for chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_images_in_dir(directory: Path, extensions: tuple[str, ...] | None = None) -> list[Path]:
    """Get all image files in a directory.

    Args:
        directory: Directory to search.
        extensions: Tuple of valid extensions. Defaults to common image formats.

    Returns:
        Sorted list of image file paths.
    """
    if extensions is None:
        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

    if not directory.exists():
        return []

    images = [
        f
        for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ]
    return sorted(images)


def get_label_path_for_image(image_path: Path) -> Path:
    """Get the corresponding label file path for an image.

    Assumes standard YOLO directory structure: images/ and labels/ are siblings.

    Args:
        image_path: Path to image file.

    Returns:
        Path to corresponding .txt label file.
    """
    label_dir = image_path.parent.parent / "labels"
    return label_dir / image_path.with_suffix(".txt").name


def compute_file_hash(file_path: Path, algorithm: str = "md5") -> str:
    """Compute hash of a file for deduplication.

    Args:
        file_path: Path to file.
        algorithm: Hash algorithm (md5, sha256).

    Returns:
        Hex digest string.
    """
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# Detector Plugin Architecture
# =============================================================================


@runtime_checkable
class DetectorProtocol(Protocol):
    """Interface that all detector implementations must satisfy.

    To add a new detector:
    1. Create a class implementing this protocol.
    2. Register it: DetectorRegistry.register("my_detector", MyDetector)
    3. Add entry to config.yaml models list.
    No changes to training/evaluation/comparison code required.
    """

    def train(
        self,
        config: PipelineConfig,
        model_config: ModelConfig,
        project_dir: Path,
    ) -> Path:
        """Train the detector.

        Args:
            config: Full pipeline configuration.
            model_config: Specific model configuration.
            project_dir: Directory to save training outputs.

        Returns:
            Path to best weights file.
        """
        ...

    def evaluate(
        self,
        weights_path: Path,
        data_yaml: Path,
        config: PipelineConfig,
    ) -> dict[str, Any]:
        """Evaluate the detector on test data.

        Args:
            weights_path: Path to trained weights.
            data_yaml: Path to data.yaml.
            config: Pipeline configuration.

        Returns:
            Dictionary of evaluation metrics.
        """
        ...

    def predict(
        self,
        weights_path: Path,
        source: str | Path,
        config: PipelineConfig,
    ) -> list[dict[str, Any]]:
        """Run inference on a source.

        Args:
            weights_path: Path to model weights.
            source: Image/video/stream source.
            config: Pipeline configuration.

        Returns:
            List of prediction dictionaries.
        """
        ...

    def export(
        self,
        weights_path: Path,
        config: PipelineConfig,
    ) -> dict[str, Any]:
        """Export model to deployment formats.

        Args:
            weights_path: Path to model weights.
            config: Pipeline configuration.

        Returns:
            Export report dictionary.
        """
        ...


class DetectorRegistry:
    """Plugin registry for object detection models.

    New detectors are registered with a name and class. The training,
    evaluation, and comparison pipelines look up detectors by name,
    so adding new models requires zero changes to core code.

    Example:
        DetectorRegistry.register("my_model", MyDetectorClass)
        detector = DetectorRegistry.get("my_model")
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, detector_class: type) -> None:
        """Register a detector class.

        Args:
            name: Unique name for the detector.
            detector_class: Class implementing DetectorProtocol.
        """
        cls._registry[name] = detector_class

    @classmethod
    def get(cls, name: str) -> type:
        """Get a registered detector class by name.

        Args:
            name: Detector name.

        Returns:
            The detector class.

        Raises:
            KeyError: If detector is not registered.
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys()) or "none"
            raise KeyError(
                f"Detector '{name}' not registered. Available: {available}"
            )
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> list[str]:
        """List all registered detector names.

        Returns:
            Sorted list of registered detector names.
        """
        return sorted(cls._registry.keys())


# =============================================================================
# Ultralytics Detector (YOLO11, RT-DETR)
# =============================================================================


class UltralyticsDetector:
    """Detector adapter for all Ultralytics-supported models.

    Handles YOLO11n, YOLO11s, RT-DETR-l, and any future Ultralytics model
    through the unified YOLO API.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize with model configuration.

        Args:
            model_config: Configuration specifying model name and pretrained weights.
        """
        self.model_config = model_config

    def train(
        self,
        config: PipelineConfig,
        model_config: ModelConfig,
        project_dir: Path,
    ) -> Path:
        """Train a model using Ultralytics API.

        Args:
            config: Full pipeline configuration.
            model_config: Specific model configuration.
            project_dir: Directory to save training outputs.

        Returns:
            Path to best weights file.
        """
        from ultralytics import YOLO

        logger = logging.getLogger("utils.UltralyticsDetector")

        model = YOLO(model_config.weights)
        tc = config.training
        ac = config.augmentation

        train_args: dict[str, Any] = {
            "data": str(config.dataset.data_yaml),
            "epochs": tc.epochs,
            "batch": tc.batch_size,
            "imgsz": config.dataset.img_size,
            "patience": tc.patience,
            "optimizer": tc.optimizer,
            "lr0": tc.lr0,
            "lrf": tc.lrf,
            "weight_decay": tc.weight_decay,
            "momentum": tc.momentum,
            "amp": tc.amp,
            "seed": tc.seed,
            "workers": tc.workers,
            "project": str(project_dir),
            "name": model_config.name,
            "exist_ok": True,
            "resume": tc.resume,
            "verbose": True,
            # Complementary augmentations
            "mosaic": ac.mosaic,
            "mixup": ac.mixup,
            "hsv_h": ac.hsv_h,
            "hsv_s": ac.hsv_s,
            "hsv_v": ac.hsv_v,
            "flipud": ac.flipud,
            "fliplr": ac.fliplr,
            "degrees": ac.degrees,
            "translate": ac.translate,
            "scale": ac.scale,
            "perspective": ac.perspective,
            "erasing": ac.erasing,
        }

        logger.info(
            "Starting training: model=%s, epochs=%d, batch=%d, imgsz=%d",
            model_config.name,
            tc.epochs,
            tc.batch_size,
            config.dataset.img_size,
        )

        model.train(**train_args)

        best_weights = project_dir / model_config.name / "weights" / "best.pt"
        if not best_weights.exists():
            raise FileNotFoundError(
                f"Training completed but best weights not found at: {best_weights}"
            )

        logger.info("Training complete. Best weights: %s", best_weights)
        return best_weights

    def evaluate(
        self,
        weights_path: Path,
        data_yaml: Path,
        config: PipelineConfig,
    ) -> dict[str, Any]:
        """Evaluate model on the test split.

        Args:
            weights_path: Path to trained weights.
            data_yaml: Path to data.yaml.
            config: Pipeline configuration.

        Returns:
            Dictionary of evaluation metrics.
        """
        from ultralytics import YOLO

        model = YOLO(str(weights_path))
        ec = config.evaluation

        results = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=config.dataset.img_size,
            conf=ec.conf_threshold,
            iou=ec.iou_threshold,
            max_det=ec.max_det,
            verbose=True,
        )

        metrics: dict[str, Any] = {
            "mAP50": float(results.box.map50),
            "mAP50_95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
            "f1": float(
                2
                * results.box.mp
                * results.box.mr
                / (results.box.mp + results.box.mr + 1e-10)
            ),
            "per_class_ap50": {
                config.dataset.names[i]: float(v)
                for i, v in enumerate(results.box.ap50)
                if i < len(config.dataset.names)
            },
            "per_class_ap": {
                config.dataset.names[i]: float(v)
                for i, v in enumerate(results.box.ap)
                if i < len(config.dataset.names)
            },
        }

        return metrics

    def predict(
        self,
        weights_path: Path,
        source: str | Path,
        config: PipelineConfig,
    ) -> list[dict[str, Any]]:
        """Run inference on a source.

        Args:
            weights_path: Path to model weights.
            source: Image, video, folder, stream, or URL.
            config: Pipeline configuration.

        Returns:
            List of prediction dictionaries.
        """
        from ultralytics import YOLO

        model = YOLO(str(weights_path))
        ic = config.inference

        results = model.predict(
            source=str(source),
            imgsz=config.dataset.img_size,
            conf=ic.conf,
            iou=ic.iou,
            max_det=ic.max_det,
            verbose=False,
        )

        predictions: list[dict[str, Any]] = []
        for result in results:
            frame_preds: list[dict[str, Any]] = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    frame_preds.append(
                        {
                            "class_id": int(box.cls[0]),
                            "class_name": config.dataset.names[int(box.cls[0])],
                            "confidence": round(float(box.conf[0]), 4),
                            "bbox": {
                                "x1": round(x1, 2),
                                "y1": round(y1, 2),
                                "x2": round(x2, 2),
                                "y2": round(y2, 2),
                            },
                        }
                    )
            predictions.append(
                {
                    "source": str(result.path) if result.path else str(source),
                    "detections": frame_preds,
                    "detection_count": len(frame_preds),
                }
            )

        return predictions

    def export(
        self,
        weights_path: Path,
        config: PipelineConfig,
    ) -> dict[str, Any]:
        """Export model to deployment formats.

        Args:
            weights_path: Path to model weights.
            config: Pipeline configuration.

        Returns:
            Export report dictionary.
        """
        from ultralytics import YOLO

        logger = logging.getLogger("utils.UltralyticsDetector")
        ec = config.export
        report: dict[str, Any] = {"exports": [], "model_path": str(weights_path)}

        # Always include PyTorch
        pt_size = weights_path.stat().st_size / (1024 * 1024)
        report["exports"].append(
            {
                "format": "pytorch",
                "success": True,
                "file_path": str(weights_path),
                "file_size_mb": round(pt_size, 2),
                "input_shape": [1, 3, config.dataset.img_size, config.dataset.img_size],
                "supported_runtimes": ["PyTorch", "Ultralytics"],
                "inference_compatible": True,
            }
        )

        model = YOLO(str(weights_path))

        for fmt in ec.formats:
            try:
                logger.info("Exporting to %s...", fmt)
                export_path = model.export(
                    format=fmt,
                    imgsz=config.dataset.img_size,
                    dynamic=ec.dynamic,
                    simplify=ec.simplify if fmt == "onnx" else False,
                    opset=ec.opset if fmt == "onnx" else None,
                    half=ec.half,
                )

                exported_file = Path(export_path) if export_path else None
                file_size = (
                    round(exported_file.stat().st_size / (1024 * 1024), 2)
                    if exported_file and exported_file.exists()
                    else 0.0
                )

                runtime_map: dict[str, list[str]] = {
                    "onnx": ["ONNX Runtime", "TensorRT", "OpenVINO", "DirectML"],
                    "torchscript": ["LibTorch", "PyTorch Mobile", "C++ Inference"],
                }

                report["exports"].append(
                    {
                        "format": fmt,
                        "success": True,
                        "file_path": str(exported_file) if exported_file else "",
                        "file_size_mb": file_size,
                        "input_shape": [
                            1,
                            3,
                            config.dataset.img_size,
                            config.dataset.img_size,
                        ],
                        "supported_runtimes": runtime_map.get(fmt, [fmt]),
                        "inference_compatible": True,
                    }
                )
                logger.info("Export to %s successful: %s", fmt, exported_file)

            except Exception as e:
                logger.error("Export to %s failed: %s", fmt, str(e))
                report["exports"].append(
                    {
                        "format": fmt,
                        "success": False,
                        "error": str(e),
                        "inference_compatible": False,
                    }
                )

        return report


# =============================================================================
# Register Default Detectors
# =============================================================================

DetectorRegistry.register("yolo11n", UltralyticsDetector)
DetectorRegistry.register("yolo11s", UltralyticsDetector)
DetectorRegistry.register("rtdetr-l", UltralyticsDetector)

# =============================================================================
# JSON Utilities
# =============================================================================


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Save data as JSON file.

    Args:
        data: Data to serialize.
        path: Output file path.
        indent: JSON indentation level.
    """
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: Path) -> Any:
    """Load data from JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed JSON data.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Progress Bar Wrapper
# =============================================================================


def progress_bar(
    iterable: Any,
    desc: str = "",
    total: int | None = None,
    unit: str = "it",
) -> tqdm:
    """Create a styled progress bar.

    Args:
        iterable: Iterable to wrap.
        desc: Description prefix.
        total: Total count (auto-detected if iterable has __len__).
        unit: Unit label for items.

    Returns:
        tqdm progress bar wrapping the iterable.
    """
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
        dynamic_ncols=True,
    )


# =============================================================================
# Resource Monitoring
# =============================================================================


def get_resource_usage() -> dict[str, Any]:
    """Get current system resource usage.

    Returns:
        Dictionary with CPU, RAM, and GPU usage statistics.
    """
    usage: dict[str, Any] = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
        "ram_percent": psutil.virtual_memory().percent,
    }

    if torch.cuda.is_available():
        usage["gpu_memory_allocated_gb"] = round(
            torch.cuda.memory_allocated() / (1024**3), 2
        )
        usage["gpu_memory_reserved_gb"] = round(
            torch.cuda.memory_reserved() / (1024**3), 2
        )
        usage["gpu_utilization"] = "N/A (requires nvidia-smi)"

    return usage
