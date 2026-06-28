"""
visualization.py — Plotting utilities for training metrics.

Generates four plots:
    1. Total Reward vs Episodes
    2. Average Customer Lifetime Value (CLV) vs Episodes
    3. Churn Rate vs Episodes
    4. Epsilon Decay Curve

All plots are saved to the 'plots/' subdirectory and also displayed.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Output directory for saved figures
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
SMOOTHING_WINDOW = 50


def _smooth(data, window=SMOOTHING_WINDOW):
    """Compute a simple rolling average for smoother curves."""
    if len(data) < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def plot_rewards(rewards, ax=None, save=True):
    """Plot total reward per episode with a smoothed trend line."""
    episodes = np.arange(1, len(rewards) + 1)
    smoothed = _smooth(rewards)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(episodes, rewards, alpha=0.25, color="#5B9BD5", label="Raw")
    ax.plot(
        np.arange(SMOOTHING_WINDOW, len(rewards) + 1),
        smoothed,
        color="#2E75B6",
        linewidth=2,
        label=f"Smoothed (window={SMOOTHING_WINDOW})",
    )
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title("Reward vs Episodes", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOT_DIR, "reward_vs_episodes.png"),
                    dpi=150, bbox_inches="tight")


def plot_clv(clvs, ax=None, save=True):
    """Plot average CLV per episode with smoothing."""
    episodes = np.arange(1, len(clvs) + 1)
    smoothed = _smooth(clvs)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(episodes, clvs, alpha=0.25, color="#70AD47", label="Raw")
    ax.plot(
        np.arange(SMOOTHING_WINDOW, len(clvs) + 1),
        smoothed,
        color="#548235",
        linewidth=2,
        label=f"Smoothed (window={SMOOTHING_WINDOW})",
    )
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Customer Lifetime Value", fontsize=12)
    ax.set_title("Average CLV vs Episodes", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOT_DIR, "clv_vs_episodes.png"),
                    dpi=150, bbox_inches="tight")


def plot_churn_rate(churns, ax=None, save=True):
    """Plot rolling churn rate over episodes."""
    smoothed = _smooth(churns)
    smoothed_pct = np.array(smoothed) * 100

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        np.arange(SMOOTHING_WINDOW, len(churns) + 1),
        smoothed_pct,
        color="#C00000",
        linewidth=2,
    )
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Churn Rate (%)", fontsize=12)
    ax.set_title("Churn Rate vs Episodes", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOT_DIR, "churn_rate_vs_episodes.png"),
                    dpi=150, bbox_inches="tight")


def plot_epsilon(epsilons, ax=None, save=True):
    """Plot the epsilon decay curve over episodes."""
    episodes = np.arange(1, len(epsilons) + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(episodes, epsilons, color="#7030A0", linewidth=2)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Epsilon (ε)", fontsize=12)
    ax.set_title("Epsilon Decay Curve", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOT_DIR, "epsilon_decay.png"),
                    dpi=150, bbox_inches="tight")


def plot_all(metrics: dict):
    """
    Generate a combined 2×2 figure of all training metrics and also
    save individual plots.

    Parameters
    ----------
    metrics : dict with keys 'rewards', 'clvs', 'churns', 'epsilons'
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Individual plots (saved automatically)
    plot_rewards(metrics["rewards"])
    plot_clv(metrics["clvs"])
    plot_churn_rate(metrics["churns"])
    plot_epsilon(metrics["epsilons"])

    # Combined 2×2 dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Customer Retention RL — Training Dashboard",
                 fontsize=16, fontweight="bold", y=1.02)

    plot_rewards(metrics["rewards"], ax=axes[0, 0], save=False)
    plot_clv(metrics["clvs"], ax=axes[0, 1], save=False)
    plot_churn_rate(metrics["churns"], ax=axes[1, 0], save=False)
    plot_epsilon(metrics["epsilons"], ax=axes[1, 1], save=False)

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "training_dashboard.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plots saved to: {PLOT_DIR}/")
