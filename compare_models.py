"""Model comparison and ranking module for PPE Detection.

Aggregates evaluation results, computes rankings, and generates comparison reports.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import pandas as pd

from utils import (
    DEFAULT_CONFIG_PATH,
    PipelineConfig,
    ensure_dir,
    load_config,
    load_json,
    save_json,
    setup_logger,
)


def normalize_series(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Normalize a pandas Series to [0, 1] range.

    Args:
        series: Numeric series to normalize.
        higher_is_better: If False, reverses the scale (1.0 = min value).

    Returns:
        Normalized series.
    """
    if series.empty or series.nunique() == 1:
        return pd.Series(1.0, index=series.index)

    min_val = series.min()
    max_val = series.max()

    norm = (series - min_val) / (max_val - min_val)
    if not higher_is_better:
        norm = 1.0 - norm

    return norm


def compare_models(config: PipelineConfig) -> bool:
    """Compare all evaluated models and generate reports.

    Args:
        config: Pipeline configuration.

    Returns:
        True if successful.
    """
    logger = setup_logger("compare_models", config)

    reports_dir = config.get_outputs_dir() / "reports"
    if not reports_dir.exists():
        logger.error(f"Reports directory not found: {reports_dir}")
        return False

    eval_files = list(reports_dir.glob("evaluation_*.json"))
    if not eval_files:
        logger.error(f"No evaluation reports found in {reports_dir}")
        return False

    logger.info(f"Found {len(eval_files)} evaluation reports.")

    # Load all reports
    data: list[dict[str, Any]] = []
    for f in eval_files:
        try:
            data.append(load_json(f))
        except Exception as e:
            logger.warning(f"Failed to load {f.name}: {e}")

    if not data:
        return False

    # Build dataframe
    df = pd.DataFrame(
        [
            {
                "Model": d.get("model_name", "unknown"),
                "mAP50": d.get("mAP50", 0.0),
                "mAP50-95": d.get("mAP50_95", 0.0),
                "Precision": d.get("precision", 0.0),
                "Recall": d.get("recall", 0.0),
                "F1": d.get("f1", 0.0),
                "FPS": d.get("fps", 0.0),
                "Latency_ms": d.get("latency_ms", 0.0),
                "Size_MB": d.get("size_mb", 0.0),
                "Params": d.get("params", 0),
            }
            for d in data
        ]
    )

    df.set_index("Model", inplace=True)

    # Ranking Logic
    # 0.5*mAP50-95 + 0.2*normalized_fps + 0.15*(1-normalized_size) + 0.15*F1
    score_map = normalize_series(df["mAP50-95"], True)
    score_fps = normalize_series(df["FPS"], True)
    score_size = normalize_series(df["Size_MB"], False)  # smaller is better
    score_f1 = normalize_series(df["F1"], True)

    df["Composite_Score"] = 0.5 * score_map + 0.2 * score_fps + 0.15 * score_size + 0.15 * score_f1

    # Sort by score
    df = df.sort_values("Composite_Score", ascending=False)

    # Save CSV
    out_dir = config.get_outputs_dir() / "comparison"
    ensure_dir(out_dir)

    csv_path = out_dir / "comparison.csv"
    df.to_csv(csv_path)
    logger.info(f"Saved tabular comparison to {csv_path}")

    # Generate JSON report
    best_model = df.index[0]
    best_metrics = df.iloc[0].to_dict()

    recommendation = (
        f"Model '{best_model}' is recommended based on the highest composite score "
        f"({best_metrics['Composite_Score']:.3f}). It balances accuracy (mAP50-95: "
        f"{best_metrics['mAP50-95']:.3f}), speed ({best_metrics['FPS']:.1f} FPS), "
        f"and model size ({best_metrics['Size_MB']:.1f} MB)."
    )

    report = {
        "models_evaluated": len(df),
        "metrics": df.to_dict(orient="index"),
        "best_model": best_model,
        "recommendation_rationale": recommendation,
    }

    json_path = out_dir / "comparison_results.json"
    save_json(report, json_path)
    logger.info(f"Saved comparison report to {json_path}")
    logger.info(f"\nRecommendation:\n{recommendation}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare evaluated models.")
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to config file"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        success = compare_models(config)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
