# Production-Grade PPE Detection Pipeline — Implementation Plan v2

## Overview

Build an enterprise-grade **Personal Protective Equipment (PPE) Detection** system around a 17,264-image YOLO dataset. The pipeline covers data validation, multi-model training, evaluation, comparison, model export, inference, visualization, REST API, Docker deployment, and CI/CD.

---

## Key Changes from v1

| Area | v1 | v2 |
|---|---|---|
| Models | YOLOv11n, YOLOv8n, RT-DETR-l | **YOLO11n, YOLO11s, RT-DETR-l** |
| Architecture | Fixed model list | **Plugin architecture** — new detectors added without modifying core |
| HPO | Enabled by default | **Disabled by default**, `--hpo` flag to enable |
| Model Export | Not included | **PT + ONNX + TorchScript** with export report |
| Docker | Not included | **Dockerfile + docker-compose.yml** (GPU/CPU) |
| CI/CD | Not included | **GitHub Actions** (Black, isort, Flake8, mypy, smoke tests) |
| API docs | Basic endpoints | **Full OpenAPI**, Pydantic schemas, request/response examples |
| Code quality | PEP8 | **Black + isort + Flake8 + mypy**, dataclasses, pathlib, logging |
| Verification | Manual | **Automated end-to-end repository verification** |
| Dataset validation | Advisory | **Strict — abort training if validation fails** |

---

## User Review Required

> [!IMPORTANT]
> **Class Names**: `data.yaml` defines 6 classes: `boots, gloves, goggles, helmet, person, vest`. This is used as the **sole** source of truth. Any differing class names from previous prompts are ignored.

> [!IMPORTANT]
> **Annotation Format**: Labels are in YOLOv8 **segmentation polygon** format. Ultralytics natively handles polygon→bbox conversion for detection tasks. No manual conversion will be performed.

> [!WARNING]
> **Plugin Architecture**: The core pipeline uses a `DetectorRegistry` pattern. YOLO11n, YOLO11s, and RT-DETR-l are registered as default detectors. Adding YOLOv8, Faster R-CNN, etc. later requires only writing a new adapter class and registering it — zero changes to training/evaluation/comparison code.

---

## Open Questions

1. **GPU Availability**: Do you have a CUDA-capable GPU? This affects batch size defaults and mixed precision. The code auto-detects, but knowing upfront helps set expectations for ~17k images.

2. **Training Duration**: Default is 100 epochs with early stopping (patience=20). Prefer a faster initial run (e.g., 30 epochs)?

3. **OneDrive Locking**: Dataset is on OneDrive. Syncing can cause file locks during training. Should we copy to a local non-synced directory, or proceed as-is?

---

## Project Structure

```
ML_Deafference/
├── archive/                         # Existing dataset (untouched)
│
├── configs/
│   └── config.yaml                  # Central configuration
│
├── data_cleaning.py                 # Data validation & cleaning
├── train_models.py                  # Multi-model training pipeline
├── compare_models.py                # Model comparison & ranking
├── evaluate_models.py               # Comprehensive evaluation
├── inference.py                     # Multi-source inference engine
├── api.py                           # FastAPI REST API
├── visualization.py                 # Publication-quality figures
├── utils.py                         # Shared utilities & registry
├── export_model.py                  # Model export (PT/ONNX/TorchScript)
│
├── models/                          # Trained model weights
├── outputs/
│   ├── predictions/
│   ├── confusion_matrix/
│   ├── feature_analysis/
│   ├── comparison/
│   ├── exports/                     # Exported models + report
│   └── reports/
├── visualizations/                  # Generated figures (PNG + SVG)
├── logs/                            # Training & inference logs
│
├── best_model.pt                    # Best model (auto-selected)
├── requirements.txt
├── README.md
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml                   # GitHub Actions CI pipeline
├── pyproject.toml                   # Black + isort + mypy config
└── setup.cfg                        # Flake8 config
```

---

## Proposed Changes

### Configuration

#### [NEW] [config.yaml](file:///c:/Users/MoHG/OneDrive/Desktop/ML_Deafference/configs/config.yaml)

Central YAML config — **all values derived from `data.yaml` at runtime**, nothing hardcoded:

```yaml
# Dataset — paths resolved relative to project root at runtime
dataset:
  data_yaml: "archive/data.yaml"        # Source of truth
  img_size: 640

# Models to benchmark
models:
  - name: "yolo11n"
    weights: "yolo11n.pt"
  - name: "yolo11s"
    weights: "yolo11s.pt"
  - name: "rtdetr-l"
    weights: "rtdetr-l.pt"

# Training
training:
  epochs: 100
  batch_size: 16
  patience: 20
  optimizer: "auto"
  lr0: 0.01
  lrf: 0.01
  weight_decay: 0.0005
  momentum: 0.937
  amp: true
  seed: 42
  workers: 8
  resume: false

# Complementary augmentation (NOT duplicating dataset pre-applied brightness/exposure)
augmentation:
  mosaic: 1.0
  mixup: 0.15
  hsv_h: 0.015        # Hue only — brightness/exposure already in dataset
  hsv_s: 0.7
  hsv_v: 0.0          # Disabled — brightness already applied
  flipud: 0.0
  fliplr: 0.5
  degrees: 0.0
  translate: 0.1
  scale: 0.5
  perspective: 0.0
  erasing: 0.4

# Hyperparameter Optimization (disabled by default)
hpo:
  enabled: false
  n_trials: 10
  metric: "metrics/mAP50-95"
  direction: "maximize"

# Evaluation
evaluation:
  conf_threshold: 0.25
  iou_threshold: 0.7
  max_det: 300

# Inference
inference:
  conf: 0.5
  iou: 0.45
  max_det: 300
  line_width: 2
  save_json: true
  save_csv: true

# Export
export:
  formats: ["torchscript", "onnx"]
  dynamic: false
  simplify: true
  opset: 17
  half: false

# API
api:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  max_file_size_mb: 50

# Logging
logging:
  level: "INFO"
  log_dir: "logs"

# Paths
paths:
  models_dir: "models"
  outputs_dir: "outputs"
  visualizations_dir: "visualizations"
```

---

### 1. Shared Utilities & Plugin Architecture

#### [NEW] [utils.py](file:///c:/Users/MoHG/OneDrive/Desktop/ML_Deafference/utils.py)

**Core infrastructure used by every module:**

```python
# Key components (all with full type hints, docstrings, dataclasses):

@dataclass
class DatasetConfig:
    """Parsed from data.yaml — source of truth."""
    train_path: Path
    val_path: Path
    test_path: Path
    nc: int
    names: list[str]

@dataclass
class ModelConfig:
    """Single model configuration."""
    name: str
    weights: str

@dataclass
class PipelineConfig:
    """Complete pipeline configuration parsed from config.yaml."""
    dataset: DatasetConfig
    models: list[ModelConfig]
    training: TrainingConfig
    # ... all sections as typed dataclasses

class DetectorRegistry:
    """Plugin registry for object detection models.
    
    New detectors are added by:
    1. Creating a class implementing the DetectorProtocol
    2. Calling DetectorRegistry.register("name", MyDetector)
    
    Zero changes to training/eval/comparison code required.
    """
    _registry: dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, detector_class: type) -> None: ...
    
    @classmethod
    def get(cls, name: str) -> type: ...
    
    @classmethod
    def list_available(cls) -> list[str]: ...

class DetectorProtocol(Protocol):
    """Interface that all detectors must implement."""
    def train(self, config: PipelineConfig) -> Path: ...
    def evaluate(self, weights: Path, data: Path) -> dict: ...
    def predict(self, weights: Path, source: str | Path) -> list: ...
    def export(self, weights: Path, formats: list[str]) -> dict: ...

class UltralyticsDetector:
    """Detector adapter for all Ultralytics models (YOLO11, RT-DETR)."""
    # Single implementation handles yolo11n, yolo11s, rtdetr-l
    # via the Ultralytics unified API

# Registered by default:
DetectorRegistry.register("yolo11n", UltralyticsDetector)
DetectorRegistry.register("yolo11s", UltralyticsDetector)
DetectorRegistry.register("rtdetr-l", UltralyticsDetector)
```

Additional utilities:
- `load_config(path: Path) -> PipelineConfig` — YAML parsing + validation
- `setup_logger(name: str, config: PipelineConfig) -> logging.Logger`
- `get_device() -> torch.device` — GPU/CPU/MPS auto-detection
- `set_seed(seed: int) -> None` — numpy + torch + random + PYTHONHASHSEED
- `Timer` context manager for profiling
- `get_model_info(weights: Path) -> ModelInfo` — size, params, FLOPs
- Progress bar wrappers (tqdm)
- All paths via `pathlib.Path`, no `os.path`

---

### 2. Data Cleaning & Validation

#### [NEW] [data_cleaning.py](file:///c:/Users/MoHG/OneDrive/Desktop/ML_Deafference/data_cleaning.py)

**Strict validation that aborts training if any check fails:**

| Check | Action on Failure |
|---|---|
| Image has no matching label | **FAIL** — log missing file |
| Label has no matching image | **FAIL** — log orphan label |
| Class ID outside `[0, nc-1]` | **FAIL** — log invalid class ID with file |
| Corrupted image (can't open) | **FAIL** — log corrupted file path |
| Empty label file | **WARN** — log (valid for background images) |
| Coordinates outside `[0, 1]` | **FAIL** — log invalid coordinates |
| `nc` mismatch with actual classes found | **FAIL** — log discrepancy |

**Statistics generated (`outputs/reports/dataset_report.json`):**
- Total images/labels per split
- Class distribution (count per class)
- Annotation count distribution (objects per image)
- Bounding box area distribution (min/max/mean/std)
- Image resolution verification (confirm all 640×640)
- Duplicate image detection (perceptual hash)

**Visualizations generated (PNG 300DPI + SVG):**
- Class distribution bar chart
- Bounding box size histogram
- Objects-per-image histogram
- Spatial heatmap of bbox centers

**CLI:**
```bash
python data_cleaning.py --config configs/config.yaml
# Exit code 0 = pass, 1 = fail (training should not proceed)
```

---

### 3. Training Pipeline

#### [NEW] [train_models.py](file:///c:/Users/MoHG/OneDrive/Desktop/ML_Deafference/train_models.py)

**Flow:**
1. Load config → validate via `data_cleaning.py` checks (abort if fail)
2. For each model in config:
   - Get detector from `DetectorRegistry`
   - Train with Ultralytics unified API
   - Save weights to `models/{model_name}/weights/`
   - Log to TensorBoard
3. After all models trained → auto-select best → copy to `best_model.pt`
4. Auto-export best model (PT/ONNX/TorchScript)

**Features:**
- Auto GPU/CPU detection + mixed precision
- Early stopping (configurable patience)
- Resume from checkpoint (`--resume`)
- Complementary augmentations only (no brightness/exposure duplication)
- Experiment naming: `{model}_{YYYYMMDD_HHMMSS}`
- Progress bars for all operations

**Optuna HPO (disabled by default):**
```bash
# Normal training
python train_models.py --config configs/config.yaml

# With HPO enabled
python train_models.py --config configs/config.yaml --hpo --n-trials 15

# Train specific models only
python train_models.py --config configs/config.yaml --models yolo11n rtdetr-l
```

**Optuna implementation:**
- `MedianPruner` for early trial termination
- Search space: lr0, lrf, batch, optimizer, weight_decay, momentum, mosaic, mixup, hsv_h, hsv_s, degrees, translate, scale, fliplr
- Best params saved to `outputs/reports/hpo_{model}.json`
- Study visualization saved to `visualizations/`

---

### 4. Evaluation

#### [NEW] [evaluate_models.py](file:///c:/Users/MoHG/OneDrive/Desktop/ML_Deafference/evaluate_models.py)

**Metrics computed:**

| Category | Metrics |
|---|---|
| Detection | mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1 |
| Per-class | AP, Precision, Recall per class |
| Performance | FPS, latency (ms/image), model size (MB), param count |
| Resources | GPU memory, CPU usage, peak RAM |
| Training | Total training time, epochs completed, best epoch |

**Plots generated (PNG 300DPI + SVG):**
- Confusion matrix (normalized + raw counts)
- PR curve (per-class + aggregate)
- Precision curve, Recall curve, F1 curve
- Confidence distribution histogram

**Error analysis:**
- Top-N false positive examples with annotated images
- Top-N false negative examples (missed detections)
- Per-class failure rate table
- Failure case images saved to `outputs/predictions/errors/`

**Output:** `outputs/reports/evaluation_{model}.json`

```bash
python evaluate_models.py --config configs/config.yaml
python evaluate_models.py --config configs/config.yaml --model models/yolo11n/weights/best.pt
```

---

### 5. Model Comparison

#### [NEW] [compare_models.py](file:///c:/Users/MoHG/OneDrive/Desktop/ML_Deafference/compare_models.py)

**Outputs:**
- `outputs/comparison/comparison_results.json`
- `outputs/comparison/comparison.csv`

**Ranking criteria:**
| Priority | Metric |
|---|---|
| Primary | mAP@0.5:0.95 |
| Secondary | Inference FPS |
| Tertiary | Model size (MB) |
| Quaternary | Latency (ms) |
| Quinary | Memory consumption |

**Comparison charts (PNG 300DPI + SVG):**
- mAP bar chart across models
- Speed vs accuracy scatter plot
- Model size comparison bar chart
- Per-class AP heatmap across models
- Radar chart (mAP, FPS, 1/size, 1/latency, 1/memory)
- Training time comparison

**Final output:** Justified recommendation with written rationale for best production model.

```bash
python compare_models.py --config configs/config.yaml
```

---

### 6. Model Export

#### [NEW] [export_model.py](file:///c:/Users/MoHG/OneDrive/Desktop/ML_Deafference/export_model.py)

**Export formats:**
| Format | Extension | Use Case |
|---|---|---|
| PyTorch | `.pt` | Native inference, fine-tuning |
| ONNX | `.onnx` | Cross-platform, ONNX Runtime, TensorRT |
| TorchScript | `.torchscript` | C++ deployment, mobile |

**Export report (`outputs/exports/export_report.json`):**
```json
{
  "model_name": "yolo11n",
  "exports": [
    {
      "format": "onnx",
      "success": true,
      "file_path": "outputs/exports/yolo11n.onnx",
      "file_size_mb": 12.4,
      "input_shape": [1, 3, 640, 640],
      "supported_runtimes": ["ONNX Runtime", "TensorRT", "OpenVINO"],
      "inference_compatible": true,
      "opset_version": 17
    }
  ],
  "timestamp": "2026-06-29T02:30:00Z"
}
```

```bash
python export_model.py --config configs/config.yaml --model best_model.pt
python export_model.py --config configs/config.yaml --model models/yolo11n/weights/best.pt --formats onnx torchscript
```

---

### 7. Visualization

#### [NEW] [visualization.py](file:///c:/Users/MoHG/OneDrive/Desktop/ML_Deafference/visualization.py)

**All figures exported as PNG (300 DPI) + SVG:**

| Figure | Source |
|---|---|
| Class distribution | Dataset stats |
| Bounding box spatial heatmap | Label coordinates |
| Bounding box size histogram | Label areas |
| Training loss curves | TensorBoard logs / results.csv |
| Validation loss curves | TensorBoard logs / results.csv |
| Precision / Recall / F1 over epochs | Training results |
| PR curves (per-class + aggregate) | Evaluation |
| Confusion matrix | Evaluation |
| Sample predictions grid | Inference output |
| Confidence histogram | Inference output |
| Model comparison bar charts | Comparison results |
| Training time comparison | Training logs |
| Inference speed comparison | Evaluation results |
| Radar chart (multi-metric) | Comparison results |

**Custom styling:** Matplotlib + seaborn with PPE-themed color palette, professional typography.

```bash
python visualization.py --config configs/config.yaml --type all
python visualization.py --config configs/config.yaml --type dataset
python visualization.py --config configs/config.yaml --type training
python visualization.py --config configs/config.yaml --type comparison
```

---

### 8. Inference Engine

#### [NEW] [inference.py](file:///c:/Users/MoHG/OneDrive/Desktop/ML_Deafference/inference.py)

**Supported sources:**

| Source | CLI Example |
|---|---|
| Single image | `--source image.jpg` |
| Multiple images | `--source img1.jpg img2.jpg` |
| Folder | `--source path/to/folder/` |
| Video | `--source video.mp4` |
| Webcam | `--source 0` |
| RTSP stream | `--source rtsp://...` |
| YouTube | `--source https://youtube.com/...` |

**Output:**
- Annotated images/frames with bboxes, labels, confidence
- FPS overlay on video/stream
- Violation count (missing PPE detection)
- JSON export: `outputs/predictions/results.json`
- CSV export: `outputs/predictions/results.csv`
- Annotated video: `outputs/predictions/output_video.mp4`

```bash
python inference.py --model best_model.pt --source archive/test/images/ --conf 0.5 --save
```

---

### 9. REST API

#### [NEW] [api.py](file:///c:/Users/MoHG/OneDrive/Desktop/ML_Deafference/api.py)

**Full OpenAPI documentation with Pydantic models:**

```python
# Pydantic schemas for all request/response types

class BoundingBox(BaseModel):
    x1: float = Field(..., description="Top-left X coordinate", example=100.0)
    y1: float = Field(..., description="Top-left Y coordinate", example=50.0)
    x2: float = Field(..., description="Bottom-right X coordinate", example=200.0)
    y2: float = Field(..., description="Bottom-right Y coordinate", example=150.0)

class Detection(BaseModel):
    class_id: int = Field(..., description="Class index", example=3)
    class_name: str = Field(..., description="Class label", example="helmet")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence", example=0.92)
    bbox: BoundingBox

class PredictionResponse(BaseModel):
    success: bool = True
    predictions: list[Detection]
    violation_count: int = Field(..., description="Number of PPE violations detected")
    processing_time_ms: float
    image_size: dict[str, int]

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    model_loaded: bool
    device: str = Field(..., example="cuda:0")
    gpu_available: bool
    model_name: str

class MetricsResponse(BaseModel):
    model_name: str
    mAP50: float
    mAP50_95: float
    inference_fps: float
    model_size_mb: float

class ClassesResponse(BaseModel):
    classes: dict[int, str]
    count: int

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: str | None = None
```

**Endpoints:**

| Endpoint | Method | Request | Response | Status Codes |
|---|---|---|---|---|
| `/predict/image` | POST | `UploadFile` (image) + optional `conf`, `iou` query params | `PredictionResponse` | 200, 400, 413, 422, 500 |
| `/predict/video` | POST | `UploadFile` (video) + optional params | `PredictionResponse` (per-frame aggregated) | 200, 400, 413, 422, 500 |
| `/predict/webcam` | POST | `conf`, `iou` params | Streaming response | 200, 400, 500 |
| `/health` | GET | — | `HealthResponse` | 200, 503 |
| `/metrics` | GET | — | `MetricsResponse` | 200, 503 |
| `/classes` | GET | — | `ClassesResponse` | 200 |

**Every endpoint includes:**
- Pydantic validation with detailed error messages
- Request/response examples in OpenAPI schema
- Proper HTTP status codes
- Exception handlers returning `ErrorResponse`
- Swagger UI auto-generated at `/docs`
- ReDoc at `/redoc`

```bash
python api.py --config configs/config.yaml --model best_model.pt --port 8000
```

---

### 10. Docker

#### [NEW] [Dockerfile](file:///c:/Users/MoHG/OneDrive/Desktop/ML_Deafference/Dockerfile)

```dockerfile
# Multi-stage build
# Stage 1: Dependencies
# Stage 2: Runtime (slim image)
# GPU support via nvidia/cuda base image
# CPU fallback via standard python image
# Non-root user for security
# Health check included
```

#### [NEW] [docker-compose.yml](file:///c:/Users/MoHG/OneDrive/Desktop/ML_Deafference/docker-compose.yml)

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./configs:/app/configs
    environment:
      - NVIDIA_VISIBLE_DEVICES=all    # GPU when available
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
# GPU
docker compose up

# CPU only
docker compose up  # auto-falls back if no GPU
```

---

### 11. CI/CD

#### [NEW] [.github/workflows/ci.yml](file:///c:/Users/MoHG/OneDrive/Desktop/ML_Deafference/.github/workflows/ci.yml)

```yaml
name: CI Pipeline
on: [push, pull_request]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Black formatting check
        run: black --check --diff .

      - name: isort import sorting check
        run: isort --check-only --diff .

      - name: Flake8 linting
        run: flake8 .

      - name: mypy type checking
        run: mypy *.py --ignore-missing-imports

      - name: Validate YAML configuration
        run: python -c "from utils import load_config; load_config('configs/config.yaml')"

      - name: Verify imports
        run: |
          python -c "import utils"
          python -c "import data_cleaning"
          python -c "import train_models"
          python -c "import evaluate_models"
          python -c "import compare_models"
          python -c "import inference"
          python -c "import visualization"
          python -c "import api"
          python -c "import export_model"

      - name: Smoke test - help flags
        run: |
          python data_cleaning.py --help
          python train_models.py --help
          python evaluate_models.py --help
          python compare_models.py --help
          python inference.py --help
          python visualization.py --help
          python export_model.py --help
```

#### [NEW] [pyproject.toml](file:///c:/Users/MoHG/OneDrive/Desktop/ML_Deafference/pyproject.toml)

```toml
[tool.black]
line-length = 99
target-version = ["py311"]

[tool.isort]
profile = "black"
line_length = 99

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
```

#### [NEW] [setup.cfg](file:///c:/Users/MoHG/OneDrive/Desktop/ML_Deafference/setup.cfg)

```ini
[flake8]
max-line-length = 99
extend-ignore = E203, W503
exclude = .git, __pycache__, .venv, archive
per-file-ignores =
    __init__.py:F401
```

---

### 12. Supporting Files

#### [NEW] [requirements.txt](file:///c:/Users/MoHG/OneDrive/Desktop/ML_Deafference/requirements.txt)

```
# Core ML
ultralytics>=8.3.0
torch>=2.0.0
torchvision>=0.15.0
onnx>=1.14.0
onnxruntime>=1.16.0

# API
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
python-multipart>=0.0.9

# Data & Config
pyyaml>=6.0
numpy>=1.24.0
pandas>=2.0.0

# Image Processing
opencv-python-headless>=4.8.0
Pillow>=10.0.0
imagehash>=4.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.13.0

# ML Utilities
scikit-learn>=1.3.0
tqdm>=4.65.0
psutil>=5.9.0

# HPO (optional, but installed for availability)
optuna>=3.6.0

# Logging
tensorboard>=2.14.0

# Video
yt-dlp>=2024.0.0

# Code Quality (dev)
black>=24.0.0
isort>=5.12.0
flake8>=7.0.0
mypy>=1.8.0
```

#### [NEW] [README.md](file:///c:/Users/MoHG/OneDrive/Desktop/ML_Deafference/README.md)

Professional README including:
- Project overview with architecture diagram (Mermaid)
- Dataset description (from Roboflow metadata, 6 classes, 17,264 images)
- Installation (pip + Docker)
- GPU requirements
- Quick start guide
- CLI usage for every script with examples
- API documentation with curl examples
- Folder structure
- Performance benchmark tables (populated after training)
- Model selection rationale
- Plugin architecture guide (how to add new detectors)
- Docker deployment guide
- CI/CD overview
- Future improvements
- References + License (MIT)

#### [NEW] [.gitignore](file:///c:/Users/MoHG/OneDrive/Desktop/ML_Deafference/.gitignore)

```gitignore
# Models & weights
*.pt
*.onnx
*.torchscript
models/
runs/

# Outputs
outputs/
visualizations/
logs/
best_model.*

# Python
__pycache__/
*.pyc
.venv/
*.egg-info/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Dataset (too large for git)
archive/
datasets/
```

---

## Verification Plan

### Automated End-to-End Verification (performed before completion)

```bash
# 1. Code quality
black --check --diff *.py
isort --check-only --diff *.py
flake8 *.py
mypy *.py --ignore-missing-imports

# 2. Import verification
python -c "import utils; import data_cleaning; import train_models; ..."

# 3. No circular imports (each module imports independently)
python -c "import utils"
python -c "import data_cleaning"
# ... (each module individually)

# 4. CLI smoke tests (--help executes without error)
python data_cleaning.py --help
python train_models.py --help
# ... (all scripts)

# 5. Config loading
python -c "from utils import load_config; c = load_config('configs/config.yaml'); print(c)"

# 6. Data validation
python data_cleaning.py --config configs/config.yaml

# 7. API startup test
# Start API, hit /health, verify 200 response
```

### Checks Performed

| Check | Method |
|---|---|
| All imports resolve | `python -c "import X"` for each module |
| No circular imports | Import each module independently |
| Every script executes | `--help` flag for each |
| Relative paths correct | Config loading + path resolution |
| Config files load | `load_config()` call |
| Model loading works | API /health endpoint |
| API starts successfully | uvicorn startup test |
| README matches implementation | Manual cross-reference |
| requirements.txt complete | All imports traced to packages |
| No TODOs/placeholders | grep verification |

**Any issues found are automatically fixed before the task is marked complete.**

---

## Execution Strategy

1. **Phase 1 — Foundation**: `utils.py` + `configs/config.yaml` + `pyproject.toml` + `setup.cfg`
2. **Phase 2 — Core Pipeline** (parallel subagents):
   - `data_cleaning.py`
   - `train_models.py`
   - `evaluate_models.py`
   - `compare_models.py`
   - `export_model.py`
3. **Phase 3 — Interface Layer** (parallel subagents):
   - `inference.py`
   - `visualization.py`
   - `api.py`
4. **Phase 4 — Deployment**: `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`
5. **Phase 5 — Documentation**: `README.md`, `requirements.txt`, `.gitignore`
6. **Phase 6 — Verification**: End-to-end automated checks + auto-fix

**Estimated scope**: ~15 files, ~6,000+ lines of production Python code.
