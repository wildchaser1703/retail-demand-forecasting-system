"""
Visualization utilities for forecasting results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def plot_forecast(
    actuals: np.ndarray,
    forecasts: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    conf_int: Optional[np.ndarray] = None,
    title: str = "Forecast vs Actual",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot forecast vs actual values.

    Parameters
    ----------
    actuals : np.ndarray
        Actual values.
    forecasts : np.ndarray
        Forecasted values.
    dates : pd.DatetimeIndex, optional
        Date index for x-axis.
    conf_int : np.ndarray, optional
        Confidence intervals (shape: (n, 2)).
    title : str, default='Forecast vs Actual'
        Plot title.
    save_path : str, optional
        Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    x = dates if dates is not None else np.arange(len(actuals))

    ax.plot(x, actuals, label="Actual", marker="o", markersize=4, linewidth=2)
    ax.plot(x, forecasts, label="Forecast", marker="s", markersize=4, linewidth=2, linestyle="--")

    if conf_int is not None:
        ax.fill_between(
            x, conf_int[:, 0], conf_int[:, 1], alpha=0.2, label="95% Confidence Interval"
        )

    ax.set_xlabel("Date" if dates is not None else "Time", fontsize=12)
    ax.set_ylabel("Sales", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_residuals(
    residuals: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Residual Analysis",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot residual analysis.

    Parameters
    ----------
    residuals : np.ndarray
        Residuals (actual - forecast).
    dates : pd.DatetimeIndex, optional
        Date index.
    title : str, default='Residual Analysis'
        Plot title.
    save_path : str, optional
        Path to save the plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = dates if dates is not None else np.arange(len(residuals))

    # Residuals over time
    axes[0, 0].plot(x, residuals, marker="o", markersize=3, linewidth=1)
    axes[0, 0].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[0, 0].set_title("Residuals Over Time", fontweight="bold")
    axes[0, 0].set_xlabel("Date" if dates is not None else "Time")
    axes[0, 0].set_ylabel("Residual")
    axes[0, 0].grid(True, alpha=0.3)

    # Histogram
    axes[0, 1].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    axes[0, 1].set_title("Residual Distribution", fontweight="bold")
    axes[0, 1].set_xlabel("Residual")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True, alpha=0.3)

    # Q-Q plot
    from scipy import stats

    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot", fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)

    # ACF plot
    from statsmodels.graphics.tsaplots import plot_acf

    plot_acf(residuals, lags=min(40, len(residuals) // 2), ax=axes[1, 1])
    axes[1, 1].set_title("Autocorrelation Function", fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_model_comparison(
    results: pd.DataFrame,
    metric: str = "mae",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot model comparison.

    Parameters
    ----------
    results : pd.DataFrame
        Results dataframe with models as index and metrics as columns.
    metric : str, default='mae'
        Metric to plot.
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save the plot.
    """
    if metric not in results.columns:
        raise ValueError(f"Metric {metric} not found in results.")

    fig, ax = plt.subplots(figsize=(10, 6))

    results_sorted = results.sort_values(metric)

    ax.barh(results_sorted.index, results_sorted[metric], color="steelblue", edgecolor="black")
    ax.set_xlabel(metric.upper(), fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.set_title(title or f"Model Comparison by {metric.upper()}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, v in enumerate(results_sorted[metric]):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot feature importance.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Dataframe with 'feature' and 'importance' columns.
    top_n : int, default=20
        Number of top features to plot.
    title : str, default='Feature Importance'
        Plot title.
    save_path : str, optional
        Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))

    top_features = importance_df.head(top_n).sort_values("importance")

    ax.barh(top_features["feature"], top_features["importance"], color="coral", edgecolor="black")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_interactive_forecast(
    actuals: np.ndarray,
    forecasts: Dict[str, np.ndarray],
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Interactive Forecast Comparison",
) -> go.Figure:
    """
    Create interactive forecast plot using Plotly.

    Parameters
    ----------
    actuals : np.ndarray
        Actual values.
    forecasts : dict
        Dictionary mapping model names to forecast arrays.
    dates : pd.DatetimeIndex, optional
        Date index.
    title : str, default='Interactive Forecast Comparison'
        Plot title.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure()

    x = dates if dates is not None else np.arange(len(actuals))

    # Add actual values
    fig.add_trace(
        go.Scatter(
            x=x,
            y=actuals,
            mode="lines+markers",
            name="Actual",
            line=dict(color="black", width=2),
            marker=dict(size=4),
        )
    )

    # Add forecasts
    colors = px.colors.qualitative.Set2
    for i, (model_name, forecast) in enumerate(forecasts.items()):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=forecast,
                mode="lines+markers",
                name=model_name,
                line=dict(color=colors[i % len(colors)], width=2, dash="dash"),
                marker=dict(size=4),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date" if dates is not None else "Time",
        yaxis_title="Sales",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )

    return fig


def plot_seasonal_decomposition(
    series: pd.Series,
    period: int = 7,
    title: str = "Seasonal Decomposition",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot seasonal decomposition.

    Parameters
    ----------
    series : pd.Series
        Time series to decompose.
    period : int, default=7
        Seasonal period.
    title : str, default='Seasonal Decomposition'
        Plot title.
    save_path : str, optional
        Path to save the plot.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    decomposition = seasonal_decompose(series, model="additive", period=period)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10))

    decomposition.observed.plot(ax=axes[0], title="Observed")
    axes[0].grid(True, alpha=0.3)

    decomposition.trend.plot(ax=axes[1], title="Trend")
    axes[1].grid(True, alpha=0.3)

    decomposition.seasonal.plot(ax=axes[2], title="Seasonal")
    axes[2].grid(True, alpha=0.3)

    decomposition.resid.plot(ax=axes[3], title="Residual")
    axes[3].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
