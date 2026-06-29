"""Training pipeline for PPE Detection.

Handles multi-model training with Ultralytics and optional Optuna HPO.
Automatically evaluates and selects the best model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import optuna

from data_cleaning import validate_dataset
from utils import (
    DEFAULT_CONFIG_PATH,
    DetectorRegistry,
    ModelConfig,
    PipelineConfig,
    UltralyticsDetector,
    ensure_dir,
    get_device,
    load_config,
    save_json,
    set_seed,
    setup_logger,
    timer,
)


def run_hpo(
    config: PipelineConfig, model_config: ModelConfig, project_dir: Path
) -> dict[str, Any]:
    """Run Optuna Hyperparameter Optimization.

    Args:
        config: Base pipeline configuration.
        model_config: Model to optimize.
        project_dir: Directory for saving results.

    Returns:
        Best hyperparameters.
    """
    logger = setup_logger("train_models", config)
    logger.info(f"Starting HPO for {model_config.name} with {config.hpo.n_trials} trials...")

    def objective(trial: optuna.Trial) -> float:
        # Define search space
        lr0 = trial.suggest_float("lr0", 1e-5, 1e-1, log=True)
        lrf = trial.suggest_float("lrf", 0.001, 0.1)
        batch = trial.suggest_categorical("batch", [8, 16, 32])
        optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        momentum = trial.suggest_float("momentum", 0.8, 0.99)

        mosaic = trial.suggest_float("mosaic", 0.0, 1.0)
        mixup = trial.suggest_float("mixup", 0.0, 0.5)
        hsv_h = trial.suggest_float("hsv_h", 0.0, 0.04)
        hsv_s = trial.suggest_float("hsv_s", 0.0, 1.0)
        degrees = trial.suggest_float("degrees", 0.0, 45.0)
        translate = trial.suggest_float("translate", 0.0, 0.3)
        scale = trial.suggest_float("scale", 0.0, 0.9)
        fliplr = trial.suggest_float("fliplr", 0.0, 1.0)

        # Modify config
        trial_config = PipelineConfig(**config.__dict__)
        trial_config.training.lr0 = lr0
        trial_config.training.lrf = lrf
        trial_config.training.batch_size = batch
        trial_config.training.optimizer = optimizer
        trial_config.training.weight_decay = weight_decay
        trial_config.training.momentum = momentum

        trial_config.augmentation.mosaic = mosaic
        trial_config.augmentation.mixup = mixup
        trial_config.augmentation.hsv_h = hsv_h
        trial_config.augmentation.hsv_s = hsv_s
        trial_config.augmentation.degrees = degrees
        trial_config.augmentation.translate = translate
        trial_config.augmentation.scale = scale
        trial_config.augmentation.fliplr = fliplr

        # Reduce epochs for HPO
        trial_config.training.epochs = max(10, config.training.epochs // 3)

        # Create a unique trial config so names don't clash
        trial_model_cfg = ModelConfig(
            name=f"{model_config.name}_trial_{trial.number}", weights=model_config.weights
        )

        try:
            detector_class = DetectorRegistry.get(
                model_config.name.split("_")[0] if "_" in model_config.name else "yolo11n"
            )
            detector = detector_class(trial_model_cfg)
            weights_path = detector.train(trial_config, trial_model_cfg, project_dir / "hpo")

            # Evaluate to get the objective metric
            eval_metrics = detector.evaluate(weights_path, config.dataset.data_yaml, trial_config)

            # Extract metric (default metrics/mAP50-95)
            # Map YOLO output names to our eval metric names
            metric_mapping = {
                "metrics/mAP50-95": "mAP50_95",
                "metrics/mAP50": "mAP50",
            }
            metric_key = metric_mapping.get(config.hpo.metric, "mAP50_95")
            return float(eval_metrics.get(metric_key, 0.0))

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()

    study = optuna.create_study(
        direction=config.hpo.direction, pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=config.hpo.n_trials)

    best_params = study.best_params
    logger.info(f"HPO finished. Best params: {best_params}")

    # Save best params
    hpo_dir = config.get_outputs_dir() / "reports"
    ensure_dir(hpo_dir)
    save_json(
        {"best_params": best_params, "best_value": study.best_value},
        hpo_dir / f"hpo_{model_config.name}.json",
    )
    return cast(dict[str, Any], best_params)


def train_pipeline(
    config: PipelineConfig, specific_models: list[str] | None = None, run_hpo_flag: bool = False
) -> None:
    """Run the main training pipeline.

    Args:
        config: Pipeline configuration.
        specific_models: List of model names to train (if None, trains all in config).
        run_hpo_flag: Whether to run HPO before training.
    """
    logger = setup_logger("train_models", config)
    set_seed(config.training.seed)

    device = get_device()
    logger.info(f"Using device: {device}")

    if not validate_dataset(config):
        logger.error("Dataset validation failed. Aborting training.")
        sys.exit(1)

    models_to_train = config.models
    if specific_models:
        models_to_train = [m for m in config.models if m.name in specific_models]

    if not models_to_train:
        logger.error(
            f"No valid models found to train. Available in config: {[m.name for m in config.models]}"
        )
        sys.exit(1)

    project_dir = config.get_models_dir()
    ensure_dir(project_dir)

    best_overall_map = -1.0
    best_overall_weights = None
    best_model_name = None

    for model_config in models_to_train:
        logger.info(f"=== Starting pipeline for {model_config.name} ===")

        train_config = PipelineConfig(**config.__dict__)

        if run_hpo_flag or config.hpo.enabled:
            best_params = run_hpo(config, model_config, project_dir)
            # Apply best params
            for k, v in best_params.items():
                if hasattr(train_config.training, k):
                    setattr(train_config.training, k, v)
                elif hasattr(train_config.augmentation, k):
                    setattr(train_config.augmentation, k, v)

        # Training
        try:
            # We determine the detector type. By default in this pipeline, all use UltralyticsDetector
            detector_class = DetectorRegistry.get(model_config.name)
        except KeyError:
            # Fallback for dynamic model names
            detector_class = UltralyticsDetector

        detector = detector_class(model_config)

        with timer(f"Training {model_config.name}") as t:
            weights_path = detector.train(train_config, model_config, project_dir)

        logger.info(f"Training completed in {t.elapsed_formatted}")

        # Evaluate to get score for model selection
        metrics = detector.evaluate(weights_path, config.dataset.data_yaml, train_config)
        current_map = metrics.get("mAP50_95", 0.0)

        logger.info(f"Model {model_config.name} achieved mAP50-95: {current_map:.4f}")

        if current_map > best_overall_map:
            best_overall_map = current_map
            best_overall_weights = weights_path
            best_model_name = model_config.name

    if best_overall_weights:
        logger.info("=== Training Complete ===")
        logger.info(f"Best model: {best_model_name} (mAP50-95: {best_overall_map:.4f})")

        # Symlink/copy best model
        import shutil

        best_dest = config.project_root / "best_model.pt"
        shutil.copy2(best_overall_weights, best_dest)
        logger.info(f"Copied best weights to {best_dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPE Detection models.")
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to config file"
    )
    parser.add_argument("--models", type=str, nargs="+", help="Specific models to train")
    parser.add_argument("--hpo", action="store_true", help="Run Optuna HPO")
    parser.add_argument("--n-trials", type=int, help="Number of HPO trials")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    args = parser.parse_args()

    try:
        config = load_config(args.config)

        if args.n_trials:
            config.hpo.n_trials = args.n_trials
        if args.resume:
            config.training.resume = True
        if args.epochs:
            config.training.epochs = args.epochs
        if args.batch_size:
            config.training.batch_size = args.batch_size

        train_pipeline(config, args.models, args.hpo)
        sys.exit(0)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
