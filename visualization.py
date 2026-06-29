"""Visualization module for PPE Detection.

Generates publication-quality charts for dataset, training, evaluation, and comparison.
Saves all figures as PNG (300 DPI) and SVG.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import (
    DEFAULT_CONFIG_PATH,
    PipelineConfig,
    ensure_dir,
    load_config,
    load_json,
    setup_logger,
)


def _setup_style() -> None:
    """Set up matplotlib/seaborn style for publication quality."""
    plt.style.use('dark_background')
    sns.set_theme(
        style="darkgrid",
        rc={
            "axes.facecolor": "#1a1a2e",
            "figure.facecolor": "#1a1a2e",
            "text.color": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "grid.color": "#2d2d44",
            "grid.alpha": 0.3,
            "axes.edgecolor": "#2d2d44"
        }
    )


def _get_palette() -> list[str]:
    """Get standard PPE color palette."""
    return ['#e94560', '#0f3460', '#16213e', '#533483', '#00b4d8', '#2a9d8f']


def _save_figure(fig: plt.Figure, name: str, output_dir: Path) -> None:
    """Save figure in both PNG and SVG formats."""
    ensure_dir(output_dir)
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output_dir / f"{name}.svg", format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_dataset_plots(config: PipelineConfig) -> None:
    """Generate dataset visualizations from report."""
    logger = setup_logger("visualization", config)
    _setup_style()
    
    report_path = config.get_outputs_dir() / "reports" / "dataset_report.json"
    if not report_path.exists():
        logger.warning(f"Dataset report not found at {report_path}")
        return
        
    report = load_json(report_path)
    viz_dir = config.get_visualizations_dir()
    
    # Class distribution (Horizontal)
    dist = report.get("class_distribution", {})
    if dist:
        fig, ax = plt.subplots(figsize=(12, 8))
        classes = list(dist.keys())
        counts = list(dist.values())
        
        # Sort by count
        sorted_idx = np.argsort(counts)
        classes = [classes[i] for i in sorted_idx]
        counts = [counts[i] for i in sorted_idx]
        
        sns.barplot(x=counts, y=classes, hue=classes, palette=_get_palette(), legend=False, ax=ax)
        ax.set_title("Class Distribution", fontsize=16, pad=20)
        ax.set_xlabel("Number of Instances")
        
        for i, v in enumerate(counts):
            ax.text(v + max(counts)*0.01, i, str(v), color='white', va='center')
            
        _save_figure(fig, "dataset_class_distribution", viz_dir)
        logger.info("Generated dataset_class_distribution")

    # Objects per image
    objs = report.get("objects_per_image", [])
    if objs:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.histplot(objs, bins=range(0, max(objs)+2) if max(objs) < 50 else 30, color='#00b4d8', ax=ax, discrete=max(objs)<50)
        ax.set_title("Objects per Image Distribution", fontsize=16, pad=20)
        ax.set_xlabel("Number of Objects")
        ax.set_ylabel("Frequency")
        _save_figure(fig, "dataset_objects_per_image", viz_dir)


def generate_training_plots(config: PipelineConfig) -> None:
    """Generate training curves from results.csv files."""
    logger = setup_logger("visualization", config)
    _setup_style()
    
    models_dir = config.get_models_dir()
    results_files = list(models_dir.glob("*/results.csv"))
    
    if not results_files:
        logger.warning(f"No training results.csv found in {models_dir}")
        return
        
    viz_dir = config.get_visualizations_dir()
    palette = sns.color_palette("husl", len(results_files))
    
    # Plot metric function
    def plot_metric(metric_cols: list[str], title: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(12, 8))
        plotted = False
        
        for i, f in enumerate(results_files):
            model_name = f.parent.name
            try:
                df = pd.read_csv(f)
                # Strip whitespace from columns
                df.columns = df.columns.str.strip()
                
                for col in metric_cols:
                    if col in df.columns:
                        ax.plot(df.index, df[col], label=f"{model_name}", color=palette[i], linewidth=2)
                        plotted = True
                        break
            except Exception as e:
                logger.warning(f"Failed to plot {f.name}: {e}")
                
        if plotted:
            ax.set_title(title, fontsize=16, pad=20)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            _save_figure(fig, filename, viz_dir)
            logger.info(f"Generated {filename}")
        else:
            plt.close(fig)

    # Losses
    plot_metric(["train/box_loss"], "Training Box Loss", "Loss", "train_box_loss")
    plot_metric(["val/box_loss"], "Validation Box Loss", "Loss", "val_box_loss")
    
    # Metrics
    plot_metric(["metrics/mAP50-95(B)", "metrics/mAP50-95"], "Validation mAP50-95", "mAP", "train_map50_95")


def generate_comparison_plots(config: PipelineConfig) -> None:
    """Generate model comparison charts."""
    logger = setup_logger("visualization", config)
    _setup_style()
    
    comp_path = config.get_outputs_dir() / "comparison" / "comparison.csv"
    if not comp_path.exists():
        logger.warning(f"Comparison CSV not found at {comp_path}")
        return
        
    df = pd.read_csv(comp_path)
    if df.empty or len(df) < 2:
        logger.warning("Not enough models in comparison CSV to generate comparison plots.")
        return
        
    viz_dir = config.get_visualizations_dir()
    
    # mAP Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(df))
    width = 0.35
    
    ax.bar(x - width/2, df["mAP50"], width, label='mAP50', color='#0f3460')
    ax.bar(x + width/2, df["mAP50-95"], width, label='mAP50-95', color='#e94560')
    
    ax.set_ylabel('Score')
    ax.set_title('mAP Comparison by Model', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    _save_figure(fig, "compare_map_bars", viz_dir)
    logger.info("Generated compare_map_bars")

    # Speed vs Accuracy
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(df["FPS"], df["mAP50-95"], s=df["Size_MB"]*5, alpha=0.7, c=range(len(df)), cmap='viridis')
    
    for i, txt in enumerate(df["Model"]):
        ax.annotate(txt, (df["FPS"].iloc[i], df["mAP50-95"].iloc[i]), xytext=(10, 10), textcoords='offset points', color='white')
        
    ax.set_title('Speed vs Accuracy (Bubble size = Model Size)', fontsize=16, pad=20)
    ax.set_xlabel('Inference Speed (FPS)')
    ax.set_ylabel('mAP50-95')
    
    _save_figure(fig, "compare_speed_vs_accuracy", viz_dir)
    logger.info("Generated compare_speed_vs_accuracy")
    
    # Radar Chart
    metrics = ["mAP50-95", "F1", "FPS", "Size_MB", "Latency_ms"]
    
    # Normalize metrics for radar [0, 1]
    radar_df = df.copy()
    for m in ["mAP50-95", "F1", "FPS"]:
        radar_df[f"norm_{m}"] = (radar_df[m] - radar_df[m].min()) / (radar_df[m].max() - radar_df[m].min() + 1e-10)
    for m in ["Size_MB", "Latency_ms"]:
        radar_df[f"norm_{m}"] = 1.0 - (radar_df[m] - radar_df[m].min()) / (radar_df[m].max() - radar_df[m].min() + 1e-10)
        
    norm_metrics = [f"norm_{m}" for m in metrics]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for i, row in radar_df.iterrows():
        values = row[norm_metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["Model"])
        ax.fill(angles, values, alpha=0.1)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title('Normalized Multi-Metric Comparison (Larger is Better)', y=1.1, fontsize=16)
    ax.legend(bbox_to_anchor=(1.1, 1.1))
    
    _save_figure(fig, "compare_radar", viz_dir)
    logger.info("Generated compare_radar")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PPE Detection visualizations.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--type", type=str, choices=["all", "dataset", "training", "evaluation", "comparison"], default="all")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        
        if args.type in ["all", "dataset"]:
            generate_dataset_plots(config)
        if args.type in ["all", "training"]:
            generate_training_plots(config)
        if args.type in ["all", "comparison"]:
            generate_comparison_plots(config)
            
        sys.exit(0)
    except Exception as e:
        print(f"Visualization error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
