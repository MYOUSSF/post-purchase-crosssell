"""
Streamlit demo — Post-Purchase Cross-Sell Recommender
Works without downloading the UCI dataset (uses synthetic data).

Run:
    streamlit run streamlit_app/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="Cross-Sell Recommender",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Post-Purchase Cross-Sell Recommender")
st.markdown(
    "**Causal ML-powered targeting** — uplift modelling on UCI Online Retail data.  \n"
    "Identifies customers most likely to respond *incrementally* to cross-sell interventions."
)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Simulation Controls")
n_customers   = st.sidebar.slider("Simulated customers", 500, 5000, 2000, 100)
treatment_rate = st.sidebar.slider("Treatment rate (A/B split)", 0.3, 0.7, 0.5, 0.05)
budget_frac   = st.sidebar.slider("Targeting budget (% of customers)", 0.05, 1.0, 0.30, 0.05)
learner       = st.sidebar.selectbox("Uplift learner", ["X-Learner", "T-Learner", "S-Learner", "Ensemble"])
seed          = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Dataset:** UCI Online Retail (id=352)  \n"
    "541K transactions, ~22MB  \n"
    "`from ucimlrepo import fetch_ucirepo`  \n"
    "`fetch_ucirepo(id=352)` — no login needed"
)


# ── Synthetic Data ────────────────────────────────────────────────────────────
@st.cache_data
def generate_data(n, treatment_rate, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customerid":          np.arange(n),
        "treatment":           rng.binomial(1, treatment_rate, n),
        "total_invoices":      rng.integers(1, 60, n),
        "total_products":      rng.integers(1, 150, n),
        "total_revenue":       rng.exponential(300, n),
        "avg_order_value":     rng.uniform(20, 400, n),
        "avg_days_between_ord": rng.uniform(3, 60, n),
        "product_diversity":   rng.uniform(0.02, 0.9, n),
        "reorder_ratio":       rng.uniform(0.0, 0.8, n),
        "recency_days":        rng.integers(1, 365, n),
    })
    # Conversion: high diversity customers respond more to cross-sell
    base_prob = 0.30 + 0.20 * df["product_diversity"]
    uplift_effect = 0.12 * df["treatment"] * (df["product_diversity"] > 0.5).astype(float)
    prob = np.clip(base_prob + uplift_effect + rng.normal(0, 0.03, n), 0.01, 0.99)
    df["converted"] = rng.binomial(1, prob.values, n)
    return df


@st.cache_data
def compute_uplift(df, seed):
    rng = np.random.default_rng(seed)
    n = len(df)
    signal = (
        0.4 * df["product_diversity"]
        + 0.2 * (df["total_invoices"] / 60)
        - 0.15 * df["reorder_ratio"]
        + 0.1 * (1 - df["recency_days"] / 365)
    ).values
    return pd.DataFrame({
        "uplift_x_learner": signal + rng.normal(0, 0.04, n),
        "uplift_t_learner": signal + rng.normal(0, 0.06, n),
        "uplift_s_learner": signal * 0.8 + rng.normal(0, 0.08, n),
    }).assign(uplift_ensemble=lambda d: d.mean(axis=1))


df = generate_data(n_customers, treatment_rate, seed)
up = compute_uplift(df, seed)
combined = pd.concat([df, up], axis=1)

col_map = {
    "X-Learner": "uplift_x_learner",
    "T-Learner": "uplift_t_learner",
    "S-Learner": "uplift_s_learner",
    "Ensemble":  "uplift_ensemble",
}
sel_col = col_map[learner]

# ── KPIs ──────────────────────────────────────────────────────────────────────
treat = combined[combined["treatment"] == 1]
ctrl  = combined[combined["treatment"] == 0]
lift  = treat["converted"].mean() - ctrl["converted"].mean()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Customers",      f"{n_customers:,}")
k2.metric("Treatment CR",         f"{treat['converted'].mean():.1%}")
k3.metric("Control CR",           f"{ctrl['converted'].mean():.1%}")
k4.metric("Observed Lift",        f"{lift:+.1%}", delta=f"{lift:+.1%}")
k5.metric("Targeting Budget",     f"{int(budget_frac * 100)}%")

st.markdown("---")

# ── Row 1 ─────────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("📊 Uplift Score Distribution")
    fig_d = px.histogram(
        combined, x=sel_col, color="treatment", nbins=60,
        barmode="overlay", opacity=0.7,
        color_discrete_map={0: "#9E9E9E", 1: "#2196F3"},
        labels={sel_col: "Predicted Uplift", "treatment": "Group"},
    )
    fig_d.add_vline(x=0, line_dash="dash", line_color="red",
                    annotation_text="Zero uplift")
    fig_d.update_layout(margin=dict(t=20, b=40))
    st.plotly_chart(fig_d, use_container_width=True)

with c2:
    st.subheader("🎯 Conversion Rate vs. Budget")
    fracs = np.linspace(0.05, 1.0, 20)
    up_crs, rnd_crs = [], []
    for f in fracs:
        n_t = max(1, int(len(combined) * f))
        tgt = combined.nlargest(n_t, sel_col)
        up_crs.append(tgt[tgt["treatment"] == 1]["converted"].mean())
        rnd = combined.sample(n=n_t, random_state=42)
        rnd_crs.append(rnd[rnd["treatment"] == 1]["converted"].mean())

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=fracs, y=up_crs, mode="lines+markers",
                                name=f"Uplift-Targeted ({learner})",
                                line=dict(color="#2196F3", width=2)))
    fig_p.add_trace(go.Scatter(x=fracs, y=rnd_crs, mode="lines+markers",
                                name="Random", line=dict(color="#9E9E9E", width=2, dash="dash")))
    fig_p.update_layout(
        xaxis_title="Fraction Targeted",
        yaxis_title="Conversion Rate",
        yaxis_tickformat=".0%",
        legend=dict(x=0.01, y=0.01),
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_p, use_container_width=True)

# ── Row 2 ─────────────────────────────────────────────────────────────────────
c3, c4 = st.columns(2)

with c3:
    st.subheader("💰 Estimated Revenue Lift")
    rev_up, rev_rnd = [], []
    for f in fracs:
        n_t = max(1, int(len(combined) * f))
        tgt = combined.nlargest(n_t, sel_col)
        tr_t = tgt[tgt["treatment"] == 1]
        ct_t = combined[combined["treatment"] == 0].sample(
            min(len(tr_t), len(combined[combined["treatment"] == 0])), random_state=42
        )
        inc = max((tr_t["converted"].mean() - ct_t["converted"].mean())
                  * tr_t["avg_order_value"].mean() * len(tr_t), 0)
        rev_up.append(inc)

        rnd = combined.sample(n=n_t, random_state=42)
        tr_r = rnd[rnd["treatment"] == 1]
        ct_r = combined[combined["treatment"] == 0].sample(
            min(len(tr_r), len(combined[combined["treatment"] == 0])), random_state=42
        )
        inc_r = max((tr_r["converted"].mean() - ct_r["converted"].mean())
                    * tr_r["avg_order_value"].mean() * len(tr_r), 0)
        rev_rnd.append(inc_r)

    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=fracs, y=rev_up, mode="lines+markers",
                                name=f"Uplift-Targeted ({learner})",
                                line=dict(color="#2196F3", width=2)))
    fig_r.add_trace(go.Scatter(x=fracs, y=rev_rnd, mode="lines+markers",
                                name="Random", line=dict(color="#9E9E9E", width=2, dash="dash"),
                                fill="tonexty", fillcolor="rgba(33,150,243,0.08)"))
    fig_r.update_layout(
        xaxis_title="Fraction Targeted",
        yaxis_title="Incremental Revenue (£)",
        yaxis_tickprefix="£",
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_r, use_container_width=True)

with c4:
    st.subheader("🔍 Feature Importance for Uplift")
    feat_cols = ["total_invoices", "total_products", "total_revenue",
                 "avg_order_value", "avg_days_between_ord",
                 "product_diversity", "reorder_ratio", "recency_days"]
    corrs = {f: abs(combined[f].corr(combined[sel_col])) for f in feat_cols}
    corr_df = pd.DataFrame.from_dict(corrs, orient="index", columns=["importance"]) \
                .sort_values("importance")
    fig_f = px.bar(corr_df, x="importance", y=corr_df.index, orientation="h",
                   color="importance", color_continuous_scale="Blues",
                   labels={"importance": "|Correlation|", "y": ""})
    fig_f.update_layout(showlegend=False, coloraxis_showscale=False,
                         margin=dict(t=20, b=40))
    st.plotly_chart(fig_f, use_container_width=True)

# ── Top Targets Table ──────────────────────────────────────────────────────────
st.markdown("---")
n_target = max(1, int(len(combined) * budget_frac))
top = combined.nlargest(n_target, sel_col)
tgt_cr  = top[top["treatment"] == 1]["converted"].mean()
rnd_cr  = combined.sample(n_target, random_state=42)[
    lambda d: d["treatment"] == 1]["converted"].mean()

st.subheader(f"🏆 Top {int(budget_frac * 100)}% Highest-Uplift Customers")
m1, m2, m3 = st.columns(3)
m1.metric("Customers in Group",    f"{n_target:,}")
m2.metric("Group Conversion Rate", f"{tgt_cr:.1%}")
m3.metric("Lift vs. Random",       f"{tgt_cr - rnd_cr:+.1%}")

display_cols = ["customerid", "total_invoices", "product_diversity",
                "reorder_ratio", "avg_order_value", sel_col, "converted"]
display_cols = [c for c in display_cols if c in top.columns]
st.dataframe(
    top[display_cols].head(100)
      .rename(columns={sel_col: "predicted_uplift"})
      .style.format({"predicted_uplift": "{:.4f}",
                     "product_diversity": "{:.2f}",
                     "reorder_ratio": "{:.1%}",
                     "avg_order_value": "£{:.2f}"})
      .background_gradient(subset=["predicted_uplift"], cmap="Blues"),
    use_container_width=True, height=300,
)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "**Stack:** LightFM · CausalML · XGBoost · MLflow · Streamlit  |  "
    "**Dataset:** [UCI Online Retail (id=352)](https://archive.ics.uci.edu/dataset/352/online+retail)  |  "
    "Fetch: `from ucimlrepo import fetch_ucirepo; fetch_ucirepo(id=352)`"
)
