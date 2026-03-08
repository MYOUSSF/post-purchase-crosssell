"""
Evaluation metrics and visualisations for the cross-sell uplift system.

Plots produced:
  1. Qini uplift curves + AUUC bar chart
  2. Targeting policy comparison (conversion rate vs. budget fraction)
  3. Uplift score distributions per learner
  4. RFM feature importance (correlation with uplift)
  5. Revenue lift simulation at various targeting budgets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

COLORS = {
    "x_learner":  "#2196F3",
    "t_learner":  "#4CAF50",
    "s_learner":  "#FF9800",
    "ensemble":   "#9C27B0",
    "random":     "#9E9E9E",
    "treat_all":  "#F44336",
}


def _save(fig, path):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Qini Uplift Curves
# ─────────────────────────────────────────────────────────────────────────────

def _qini_curve(df, uplift_col, treatment_col="treatment", outcome_col="converted"):
    df = df.sort_values(uplift_col, ascending=False).reset_index(drop=True)
    n = len(df)
    n_treat = df[treatment_col].sum()
    curve = []
    for i in range(1, n + 1):
        sub = df.iloc[:i]
        tr = sub[sub[treatment_col] == 1]
        ct = sub[sub[treatment_col] == 0]
        p_t = tr[outcome_col].mean() if len(tr) > 0 else 0
        p_c = ct[outcome_col].mean() if len(ct) > 0 else 0
        curve.append((p_t - p_c) * (len(tr) / n_treat if n_treat else 0))
    return np.arange(1, n + 1) / n, np.array(curve)


def plot_uplift_curves(
    test_df: pd.DataFrame,
    uplift_df: pd.DataFrame,
    save_path: str = None,
):
    merged = pd.concat(
        [test_df.reset_index(drop=True), uplift_df.reset_index(drop=True)], axis=1
    ).loc[:, lambda d: ~d.columns.duplicated()]

    learner_map = {
        "X-Learner":  "uplift_x_learner",
        "T-Learner":  "uplift_t_learner",
        "S-Learner":  "uplift_s_learner",
        "Ensemble":   "uplift_ensemble",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Qini curves ---
    ax = axes[0]
    for label, col in learner_map.items():
        if col not in merged.columns:
            continue
        x, y = _qini_curve(merged, col)
        ax.plot(x, y, label=label, linewidth=2,
                color=COLORS.get(col.replace("uplift_", ""), "#607D8B"))

    overall_lift = (
        merged[merged["treatment"] == 1]["converted"].mean()
        - merged[merged["treatment"] == 0]["converted"].mean()
    )
    ax.plot([0, 1], [0, overall_lift], "--", color=COLORS["random"],
            label="Random", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title("Qini Uplift Curves", fontweight="bold")
    ax.set_xlabel("Fraction of Customers Targeted")
    ax.set_ylabel("Cumulative Incremental Conversion Rate")
    ax.legend(fontsize=9)

    # --- AUUC bar chart ---
    ax2 = axes[1]
    auuc_scores = {}
    for label, col in learner_map.items():
        if col not in merged.columns:
            continue
        x, y = _qini_curve(merged, col)
        auuc_scores[label] = np.trapz(y, x)

    colors = [COLORS.get(c.replace("uplift_", ""), "#607D8B")
              for c in learner_map.values() if c in merged.columns]
    bars = ax2.barh(list(auuc_scores.keys()), list(auuc_scores.values()),
                    color=colors)
    ax2.set_title("AUUC by Model", fontweight="bold")
    ax2.set_xlabel("Area Under Uplift Curve")
    for bar, val in zip(bars, auuc_scores.values()):
        ax2.text(bar.get_width() + 0.00005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.5f}", va="center", fontsize=9)

    plt.tight_layout()
    _save(fig, save_path)
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Targeting Policy Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_targeting_policy(
    test_df: pd.DataFrame,
    uplift_df: pd.DataFrame,
    budget_fractions=None,
    save_path: str = None,
):
    fracs = budget_fractions or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    merged = pd.concat(
        [test_df.reset_index(drop=True), uplift_df.reset_index(drop=True)], axis=1
    ).loc[:, lambda d: ~d.columns.duplicated()]

    ctrl_cr = merged[merged["treatment"] == 0]["converted"].mean()
    records = []

    for frac in fracs:
        n_t = int(len(merged) * frac)

        if "uplift_x_learner" in merged.columns:
            targeted = merged.nlargest(n_t, "uplift_x_learner")
            cr = targeted[targeted["treatment"] == 1]["converted"].mean()
            records.append({"fraction": frac, "policy": "Uplift-Targeted (X-Learner)", "cr": cr})

        rand = merged.sample(n=n_t, random_state=42)
        records.append({
            "fraction": frac,
            "policy": "Random",
            "cr": rand[rand["treatment"] == 1]["converted"].mean(),
        })
        records.append({"fraction": frac, "policy": "No Treatment", "cr": ctrl_cr})

    rdf = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(10, 5))
    for policy in rdf["policy"].unique():
        sub = rdf[rdf["policy"] == policy]
        style = "--" if policy == "No Treatment" else "-"
        ax.plot(sub["fraction"], sub["cr"], marker="o", linestyle=style,
                label=policy, linewidth=2)

    ax.set_title("Targeting Policy: Conversion Rate vs. Budget Fraction",
                 fontweight="bold")
    ax.set_xlabel("Fraction of Customers Targeted")
    ax.set_ylabel("Conversion Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax.legend()
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Uplift Score Distributions
# ─────────────────────────────────────────────────────────────────────────────

def plot_uplift_distributions(uplift_df: pd.DataFrame, save_path: str = None):
    learners = ["x_learner", "t_learner", "s_learner"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, learner in zip(axes, learners):
        col = f"uplift_{learner}"
        if col not in uplift_df.columns:
            ax.set_visible(False)
            continue
        ax.hist(uplift_df[col], bins=50,
                color=COLORS[learner], edgecolor="white", alpha=0.85)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
        median = uplift_df[col].median()
        ax.axvline(median, color="orange", linewidth=1.2,
                   label=f"Median: {median:.4f}")
        ax.set_title(f"{learner.replace('_', '-').title()}", fontweight="bold")
        ax.set_xlabel("Estimated Uplift")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    plt.suptitle("Uplift Score Distributions by Learner", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    test_df: pd.DataFrame,
    uplift_df: pd.DataFrame,
    feature_cols: list,
    save_path: str = None,
):
    merged = pd.concat(
        [test_df.reset_index(drop=True), uplift_df.reset_index(drop=True)], axis=1
    ).loc[:, lambda d: ~d.columns.duplicated()]

    if "uplift_ensemble" not in merged.columns:
        return

    corrs = {
        f: abs(merged[f].corr(merged["uplift_ensemble"]))
        for f in feature_cols if f in merged.columns
    }
    corr_df = (
        pd.DataFrame.from_dict(corrs, orient="index", columns=["importance"])
        .sort_values("importance")
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(corr_df.index, corr_df["importance"],
                   color="#2196F3", edgecolor="white")
    ax.set_title("Feature Correlation with Ensemble Uplift Score",
                 fontweight="bold")
    ax.set_xlabel("|Pearson Correlation|")
    for bar, val in zip(bars, corr_df["importance"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Revenue Lift Simulation
# ─────────────────────────────────────────────────────────────────────────────

def plot_revenue_lift(
    test_df: pd.DataFrame,
    uplift_df: pd.DataFrame,
    avg_order_value_col: str = "avg_order_value",
    budget_fractions=None,
    save_path: str = None,
):
    """
    Estimate incremental revenue from uplift-targeted cross-sell
    vs. random targeting at various budget fractions.
    """
    fracs = budget_fractions or np.linspace(0.05, 1.0, 20)
    merged = pd.concat(
        [test_df.reset_index(drop=True), uplift_df.reset_index(drop=True)], axis=1
    ).loc[:, lambda d: ~d.columns.duplicated()]

    if avg_order_value_col not in merged.columns:
        logger.warning("avg_order_value column not found — skipping revenue plot.")
        return None

    uplift_rev, random_rev = [], []
    for frac in fracs:
        n_t = max(1, int(len(merged) * frac))

        # Uplift targeted
        if "uplift_x_learner" in merged.columns:
            targeted = merged.nlargest(n_t, "uplift_x_learner")
            treat_sub = targeted[targeted["treatment"] == 1]
            ctrl_sub = merged[merged["treatment"] == 0].sample(
                n=min(len(treat_sub), len(merged[merged["treatment"] == 0])),
                random_state=42,
            )
            incremental = (
                treat_sub["converted"].mean() - ctrl_sub["converted"].mean()
            ) * treat_sub[avg_order_value_col].mean() * len(treat_sub)
            uplift_rev.append(max(incremental, 0))

        # Random
        rand = merged.sample(n=n_t, random_state=42)
        rand_treat = rand[rand["treatment"] == 1]
        rand_ctrl = merged[merged["treatment"] == 0].sample(
            n=min(len(rand_treat), len(merged[merged["treatment"] == 0])),
            random_state=42,
        )
        inc_rand = (
            rand_treat["converted"].mean() - rand_ctrl["converted"].mean()
        ) * rand_treat[avg_order_value_col].mean() * len(rand_treat)
        random_rev.append(max(inc_rand, 0))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fracs, uplift_rev, color="#2196F3", linewidth=2,
            label="Uplift-Targeted (X-Learner)", marker="o", markersize=4)
    ax.plot(fracs, random_rev, color="#9E9E9E", linewidth=2,
            label="Random", marker="s", markersize=4, linestyle="--")
    ax.set_title("Estimated Incremental Revenue vs. Targeting Budget",
                 fontweight="bold")
    ax.set_xlabel("Fraction of Customers Targeted")
    ax.set_ylabel("Incremental Revenue (£)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"£{v:,.0f}"))
    ax.legend()
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Summary Table
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics_table(metrics: dict) -> None:
    print("\n" + "═" * 55)
    print("  UPLIFT MODEL EVALUATION")
    print("═" * 55)
    for k, v in sorted(metrics.items()):
        print(f"  {k:<38} {v:.6f}")
    print("═" * 55 + "\n")
