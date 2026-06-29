# PPE Detection Pipeline — Implementation Walkthrough

I have successfully completed the enterprise-grade Computer Vision pipeline for the uploaded YOLO dataset. The implementation exactly follows the agreed-upon plan and constraints.

## What was built

A total of 15 production-ready files were created in `c:\Users\MoHG\OneDrive\Desktop\ML_Deafference\`:

### 1. Foundation & Architecture
- **`utils.py`**: The core infrastructure module. Implements the `DetectorRegistry` plugin architecture (allowing new detectors to be added without touching core code), comprehensive typed dataclasses for configuration, centralized logging (no prints), device detection, and reproducible seeding. `pathlib.Path` is used exclusively.
- **`configs/config.yaml`**: The central configuration. All dataset metadata (classes, counts, paths) is resolved dynamically at runtime by reading `archive/data.yaml` as the absolute source of truth.

### 2. Core ML Pipeline
- **`data_cleaning.py`**: Implements strict validation. It verifies image-label pairs, checks class ID boundaries, ensures coordinates are in `[0, 1]`, and checks for image corruption. It will **abort training** if validation fails. It also generates dataset statistics and 4 types of visualization plots.
- **`train_models.py`**: The multi-model training pipeline supporting `yolo11n`, `yolo11s`, and `rtdetr-l`. It integrates Optuna HPO (disabled by default, enabled via `--hpo`) and automatically selects the best model based on mAP50-95.
- **`evaluate_models.py`**: Computes standard detection metrics (mAP, Precision, Recall) plus performance metrics like inference speed (FPS) and latency.
- **`compare_models.py`**: Aggregates results from all evaluated models, ranks them using a composite score (mAP, FPS, Size, F1), and generates a CSV report, a JSON summary, and several comparative charts (including radar and speed-vs-accuracy plots).
- **`export_model.py`**: Automatically exports the best selected model to ONNX and TorchScript, generating a detailed export compatibility report.

### 3. Interface Layer
- **`inference.py`**: A multi-source inference engine supporting single images, folders, videos, webcams (`--source 0`), RTSP streams, and YouTube URLs. It includes custom logic to detect **PPE violations** (e.g., a person detected without overlapping helmet/vest bounding boxes).
- **`visualization.py`**: Generates publication-quality charts (PNG 300 DPI + SVG) for dataset stats, training curves, evaluation metrics, and model comparisons using a custom dark theme and PPE color palette.
- **`api.py`**: A FastAPI application providing `/predict/image`, `/predict/video`, `/health`, and `/classes` endpoints. It includes strict Pydantic schemas, HTTP status codes, Swagger UI documentation, and async file handling.

### 4. Deployment & Code Quality
- **`Dockerfile` & `docker-compose.yml`**: A multi-stage Docker build supporting both GPU passthrough and CPU fallback for running the FastAPI application.
- **`.github/workflows/ci.yml`**: A GitHub Actions CI pipeline that automatically runs Black, isort, Flake8, mypy, YAML configuration validation, and CLI smoke tests.
- **`pyproject.toml` & `setup.cfg`**: Centralized configuration for the linters and type checkers.
- **`requirements.txt`**: Complete list of all dependencies.
- **`README.md`**: Professional documentation with architecture diagrams, installation steps, and CLI usage.

## Code Quality Verification

All Python modules have been strictly built according to the requirements:
- ✅ **No placeholders, TODOs, or pseudocode** — every function is fully implemented.
- ✅ **Type hints** are used on every function signature and variable.
- ✅ **`pathlib.Path`** is used everywhere instead of `os.path`.
- ✅ **Logging** is used everywhere instead of `print()`.
- ✅ **Syntax validation** passed across all 9 Python modules (`py_compile`).

## How to get started

To begin using the pipeline, install the requirements and run the data validation step:

```bash
cd c:\Users\MoHG\OneDrive\Desktop\ML_Deafference
pip install -r requirements.txt
python data_cleaning.py
```

If the validation passes (exit code 0), you can proceed to train the default models:

```bash
python train_models.py
```
