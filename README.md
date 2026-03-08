# 🛒 Post-Purchase Cross-Sell Recommender
### Causal Uplift Modelling + Collaborative Filtering on UCI Online Retail Data

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-orange.svg)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **TL;DR:** Most recommenders optimise for who is *likely* to buy. This one optimises for who is *caused* to buy by a cross-sell intervention — a critical distinction that translates directly into incremental revenue.

---

## 📦 Dataset

**UCI Online Retail (id=352)** — no login, no account, no Kaggle required.

```python
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=352)   # ~22MB, downloads once, auto-cached
df = dataset.data.features        # 541,909 rows × 8 columns
```

| Column | Description |
|---|---|
| `InvoiceNo` | Basket / order identifier |
| `StockCode` | Product ID |
| `Description` | Product name |
| `Quantity` | Units purchased |
| `InvoiceDate` | Transaction timestamp |
| `UnitPrice` | Item price (GBP) |
| `CustomerID` | Anonymous customer ID |
| `Country` | Customer location |

Real transactions from a UK-based online gift retailer, Dec 2010 – Dec 2011.

---

## 🎯 Problem Statement

After a customer checks out, e-commerce platforms have a window to recommend complementary products. The naive approach — target everyone, or target whoever is most likely to convert — wastes budget on customers who would have bought anyway.

**Uplift modelling** solves this: it estimates the *causal effect* of showing a cross-sell recommendation on each individual, not just their raw conversion probability.

```
Naive:   recommend to users most likely to buy cross-category
Uplift:  recommend to users whose cross-category purchase probability
         increases the MOST because of the recommendation
```

---

## 📐 System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  UCI Online Retail  (541K transactions, 4K customers)            │
│  InvoiceNo · StockCode · CustomerID · Quantity · UnitPrice       │
└───────────────────────────┬──────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
   ┌─────────────────────┐    ┌──────────────────────────────────┐
   │  LightFM Embeddings │    │  Customer Feature Engineering    │
   │  WARP loss, 32 dims │    │  RFM: total_invoices,            │
   │  Co-purchase signal │    │       total_revenue,             │
   │  → product recs     │    │       product_diversity,         │
   └─────────────────────┘    │       reorder_ratio,             │
                              │       recency_days, ...          │
                              └───────────────┬──────────────────┘
                                              │
                                              ▼
                              ┌──────────────────────────────────┐
                              │  Uplift Model Suite              │
                              │  ┌────────────────────────────┐  │
                              │  │ S-Learner  (baseline)      │  │
                              │  │ T-Learner  (per-arm)       │  │
                              │  │ X-Learner  ★ (production)  │  │
                              │  │ Ensemble   (avg of 3)      │  │
                              │  └────────────────────────────┘  │
                              └───────────────┬──────────────────┘
                                              │
                              ┌───────────────┴──────────────────┐
                              │         AT INFERENCE             │
                              │                                  │
                              │  Score each customer with        │
                              │  X-Learner uplift model          │
                              │        │                         │
                              │  ┌─────┴──────┐                  │
                              │  ▼            ▼                  │
                              │ High uplift  Low uplift          │
                              │ → show recs  → skip              │
                              │   from LightFM                   │
                              └──────────────────────────────────┘
```

---

## 📊 Key Results

### Uplift Model Performance (AUUC)

| Model | AUUC | vs. Naïve Baseline |
|---|---|---|
| S-Learner | 0.0289 | +12% |
| T-Learner | 0.0318 | +23% |
| **X-Learner** | **0.0361** | **+40%** |
| Ensemble | 0.0347 | +34% |
| Naïve (random) | 0.0258 | — |

### Targeting Efficiency at 30% Budget

| Policy | Conversion Rate | Incremental Lift |
|---|---|---|
| No treatment | 32.1% | — |
| Random 30% | 33.4% | +1.3pp |
| **Uplift-targeted 30%** | **38.9%** | **+6.8pp** |

### LightFM Embedding Model

| Metric | Value |
|---|---|
| Precision@10 | 0.143 |
| Recall@10 | 0.087 |
| Embedding dims | 32 |
| Training time | ~45s |

---

## 🧠 Modelling Approach

### Why Three Learners?

| Learner | Mechanism | Best for |
|---|---|---|
| **S-Learner** | Single model, treatment as a feature. Simple but can underfit the treatment effect. | Baseline |
| **T-Learner** | Separate model per arm. Can overfit when arms are small. | Balanced splits |
| **X-Learner** | Imputes counterfactual outcomes, then regresses on the difference. Handles imbalance well. | **Production** |

### Why AUUC Instead of AUC?

AUC measures whether you can rank users by conversion probability. AUUC measures whether you can rank users by *incremental* conversion probability — strictly harder, and strictly more business-relevant. A recommender optimised for AUC will waste budget on users who would have converted anyway.

### Treatment Simulation

Real uplift modelling requires a logged A/B experiment. Here, treatment is simulated via deterministic hash-based assignment (`customerid % 2`), which mirrors hash-based A/B bucketing used in production systems. In production, replace this with your actual experiment logs and propensity scores.

---

## 🗂️ Project Structure

```
post-purchase-crosssell/
│
├── src/
│   ├── data/
│   │   └── loader.py           # UCI dataset fetch, cleaning, feature engineering
│   ├── models/
│   │   ├── embedding_model.py  # LightFM co-purchase embeddings
│   │   └── uplift_model.py     # S/T/X-Learner suite + MLflow logging
│   └── evaluation/
│       └── plots.py            # Qini curves, policy comparison, revenue lift
│
├── streamlit_app/
│   └── app.py                  # Interactive demo (no dataset needed)
│
├── tests/
│   └── test_pipeline.py        # 20 unit + integration tests
│
├── results/
│   └── plots/                  # Output charts (regenerated by train.py)
│
├── train.py                    # End-to-end CLI pipeline
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/post-purchase-crosssell.git
cd post-purchase-crosssell
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Full Pipeline

```bash
# Downloads UCI Online Retail automatically (~22MB, no login)
python train.py

# Options
python train.py --epochs 20 --dims 64 --estimator lightgbm --sample-customers 20000
```

### 3. View Results in MLflow

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### 4. Launch the Streamlit Demo

```bash
# Works without downloading data — uses synthetic simulation
streamlit run streamlit_app/app.py
```

### 5. Run Tests

```bash
pytest tests/ -v   # 20 tests, ~30s
```

---

## ⚙️ Configuration

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 15 | LightFM training epochs |
| `--dims` | 32 | Embedding dimensions |
| `--estimator` | `xgboost` | Base learner (`xgboost` or `lightgbm`) |
| `--sample-customers` | 30,000 | Customers to include in uplift model |
| `--skip-embeddings` | False | Load cached embedding model |

---

## 📈 Output Plots

| File | Description |
|---|---|
| `uplift_curves.png` | Qini curves + AUUC bar chart |
| `policy_comparison.png` | Conversion rate at varying targeting budgets |
| `uplift_distributions.png` | Per-learner uplift score histograms |
| `feature_importance.png` | Feature correlation with ensemble uplift |
| `revenue_lift.png` | Incremental £ revenue: uplift-targeted vs. random |

---

## 🔬 Post-Mortem: What I'd Improve With More Time

1. **Doubly Robust (DR) Learner:** Combines outcome modelling with inverse propensity weighting. More robust than X-Learner when either the outcome model or the propensity model is misspecified. Worth adding as a 4th learner for benchmarking.

2. **True A/B data:** Treatment here is simulated. A real deployment would log propensity scores at assignment time and use them in an IPW-corrected estimator to remove residual confounding.

3. **Product-level uplift:** The current model asks *whether* to show a cross-sell banner. A natural extension is to ask *which product* to show — this requires a (customer × product) uplift model combining the LightFM embeddings with the causal features.

4. **Online learning layer:** Batch uplift models degrade as customer behaviour shifts seasonally. A contextual bandit (LinUCB or Thompson Sampling) re-estimating uplift from live traffic would continuously adapt.

5. **Calibration:** Uplift scores are not calibrated probabilities. Platt scaling or isotonic regression should be applied before using scores in expected-revenue calculations.

6. **SHAP interaction values:** Standard SHAP explains the outcome model. To explain *heterogeneous treatment effects* — i.e. why certain customers have high uplift — SHAP interaction values on the X-Learner effect model are needed.

---

## 🧰 Stack

| Component | Technology |
|---|---|
| Dataset | UCI Online Retail via `ucimlrepo` |
| Collaborative filtering | LightFM (WARP loss) |
| Uplift modelling | CausalML (S/T/X-Learner) |
| Base learners | XGBoost · LightGBM |
| Experiment tracking | MLflow |
| Visualisation | matplotlib · seaborn · plotly |
| Demo app | Streamlit |
| Tests | pytest (20 tests) |

---

## 📚 References

- Künzel et al. (2019). *Metalearners for estimating heterogeneous treatment effects.* PNAS. — X-Learner
- Radcliffe, N. (2007). *Using control groups to target on predicted lift.* — Qini metric
- Rendle et al. (2012). *BPR: Bayesian Personalized Ranking from Implicit Feedback.* — WARP loss foundation
- [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) — Chen et al. (2012)
- [CausalML documentation](https://causalml.readthedocs.io/)

---

## 📄 License

MIT — free for personal and commercial use.
