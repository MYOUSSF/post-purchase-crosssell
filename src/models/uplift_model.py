"""
Uplift modelling for post-purchase cross-sell targeting.

Implements three meta-learners and compares them using AUUC
(Area Under the Uplift Curve):

  S-Learner  — single model, treatment as a feature (baseline)
  T-Learner  — separate model per arm
  X-Learner  — imputed treatment effects (best for imbalanced splits)

The predicted uplift score for each customer represents the estimated
*incremental* probability of a cross-category purchase caused by
showing a cross-sell recommendation — not just the raw probability.

Customers with the highest uplift scores are the optimal targets.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from causalml.inference.meta import BaseSClassifier, BaseTClassifier, BaseXClassifier
from causalml.metrics import auuc_score
import mlflow
import joblib
import logging
import os

logger = logging.getLogger(__name__)


FEATURE_COLS = [
    "total_invoices",
    "total_products",
    "total_revenue",
    "avg_order_value",
    "avg_days_between_ord",
    "product_diversity",
    "reorder_ratio",
    "recency_days",
]


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Stratified train/test split preserving treatment balance."""
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["treatment"],
    )


class UpliftModelSuite:
    """
    Trains S-, T-, and X-Learner uplift models and exposes a unified
    predict / evaluate interface.
    """

    def __init__(self, base_estimator: str = "xgboost"):
        self.base_estimator = base_estimator
        self.models: dict = {}
        self.feature_cols: list = FEATURE_COLS
        self.is_fitted = False

    # ── Base learner factory ──────────────────────────────────────────────────

    def _learner(self):
        if self.base_estimator == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
        return lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        train: pd.DataFrame,
        feature_cols: list = None,
    ) -> "UpliftModelSuite":
        self.feature_cols = feature_cols or FEATURE_COLS
        X = train[self.feature_cols].values
        T = train["treatment"].values
        Y = train["converted"].values

        logger.info("Training S-Learner...")
        self.models["s_learner"] = BaseSClassifier(learner=self._learner())
        self.models["s_learner"].fit(X=X, treatment=T, y=Y)

        logger.info("Training T-Learner...")
        self.models["t_learner"] = BaseTClassifier(
            learner=self._learner(),
            control_learner=self._learner(),
        )
        self.models["t_learner"].fit(X=X, treatment=T, y=Y)

        logger.info("Training X-Learner...")
        self.models["x_learner"] = BaseXClassifier(
            outcome_learner=self._learner(),
            effect_learner=self._learner(),
        )
        self.models["x_learner"].fit(X=X, treatment=T, y=Y)

        self.is_fitted = True
        logger.info("All uplift models trained.")
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_uplift(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return per-customer uplift scores from all three learners
        plus an ensemble average.
        """
        self._check_fitted()
        X = df[self.feature_cols].values
        out = {}

        for name, model in self.models.items():
            raw = model.predict(X=X)
            out[f"uplift_{name}"] = raw[:, 1] if raw.ndim == 2 else raw

        result = pd.DataFrame(out)
        uplift_cols = [c for c in result.columns if c.startswith("uplift_")]
        result["uplift_ensemble"] = result[uplift_cols].mean(axis=1)

        if "customerid" in df.columns:
            result.insert(0, "customerid", df["customerid"].values)

        return result

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(
        self,
        test: pd.DataFrame,
        log_mlflow: bool = True,
    ) -> dict:
        """AUUC for each learner vs. a naïve random-targeting baseline."""
        self._check_fitted()
        X = test[self.feature_cols].values
        T = test["treatment"].values
        Y = test["converted"].values
        outcome_df = pd.DataFrame({"treatment": T, "converted": Y})

        metrics = {}
        for name, model in self.models.items():
            raw = model.predict(X=X)
            uplift = raw[:, 1] if raw.ndim == 2 else raw
            auuc = auuc_score(outcome_df, uplift)
            metrics[f"auuc_{name}"] = float(auuc)
            logger.info(f"  {name} AUUC: {auuc:.5f}")

        # Naïve baseline: random ordering
        naive = auuc_score(outcome_df, T.astype(float))
        metrics["auuc_naive"] = float(naive)
        logger.info(f"  naive AUUC: {naive:.5f}")

        if log_mlflow:
            with mlflow.start_run(run_name="uplift_eval", nested=True):
                mlflow.log_metrics(metrics)
                mlflow.log_param("base_estimator", self.base_estimator)

        return metrics

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Uplift suite saved → {path}")

    @classmethod
    def load(cls, path: str) -> "UpliftModelSuite":
        return joblib.load(path)

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict/evaluate.")
