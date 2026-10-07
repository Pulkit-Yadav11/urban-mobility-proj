"""
All visualisation routines. Each function saves to outputs/plots/ and
optionally returns the figure for notebook use.
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe in scripts / CI)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns

from config import PLOTS_DIR

sns.set_theme(style="whitegrid", palette="muted")


def _save(fig: plt.Figure, name: str) -> None:
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 1. Actual vs Predicted (7-day window) ──────────────────────────────────

def plot_actual_vs_predicted(
    y_test: pd.Series,
    preds: np.ndarray,
    model_name: str,
    days: int = 7,
) -> None:
    n = days * 24
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_test.index[:n], y_test.values[:n], label="Actual", lw=1.5)
    ax.plot(y_test.index[:n], preds[:n],          label="Predicted",
            lw=1.5, linestyle="--", alpha=0.85)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    ax.set_title(f"Actual vs Predicted — {model_name} (first {days} days of test)")
    ax.set_ylabel("Traffic Volume (vehicles/h)")
    ax.legend()
    _save(fig, f"actual_vs_predicted_{model_name.replace(' ', '_')}")


# ── 2. Feature importance ──────────────────────────────────────────────────

def plot_feature_importance(
    importance: pd.Series,
    model_name: str,
    top_n: int = 20,
) -> None:
    data = importance.head(top_n)
    fig, ax = plt.subplots(figsize=(9, top_n * 0.35 + 1))
    sns.barplot(x=data.values, y=data.index, ax=ax, orient="h")
    ax.set_title(f"Feature Importance — {model_name} (top {top_n})")
    ax.set_xlabel("Importance Score")
    _save(fig, f"feature_importance_{model_name.replace(' ', '_')}")


# ── 3. Metrics comparison bar chart ───────────────────────────────────────

def plot_metrics_comparison(metrics_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        sns.barplot(x=metrics_df.index, y=metrics_df[metric], ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Model Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, "model_comparison")


# ── 4. Residuals distribution ──────────────────────────────────────────────

def plot_residuals(
    y_test: pd.Series,
    preds: np.ndarray,
    model_name: str,
) -> None:
    residuals = y_test.values - preds
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(residuals, bins=60, edgecolor="white")
    axes[0].axvline(0, color="red", lw=1.5, linestyle="--")
    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("Residual (actual − predicted)")

    # Predicted vs Residual scatter
    axes[1].scatter(preds, residuals, alpha=0.15, s=6)
    axes[1].axhline(0, color="red", lw=1.5, linestyle="--")
    axes[1].set_title("Predicted vs Residual")
    axes[1].set_xlabel("Predicted Volume")
    axes[1].set_ylabel("Residual")

    fig.suptitle(f"Residual Analysis — {model_name}", fontsize=13)
    plt.tight_layout()
    _save(fig, f"residuals_{model_name.replace(' ', '_')}")
