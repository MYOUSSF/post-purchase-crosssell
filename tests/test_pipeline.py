"""
Unit and integration tests for the Post-Purchase Cross-Sell pipeline.

Run:
    pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader import (
    clean,
    build_user_product_matrix,
    build_customer_features,
    build_treatment_dataset,
    FEATURE_COLS,
)
from src.models.uplift_model import UpliftModelSuite, split_dataset


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_retail_df():
    """Minimal Online Retail–schema DataFrame."""
    rng = np.random.default_rng(0)
    n = 800
    return pd.DataFrame({
        "InvoiceNo":   [str(i) for i in rng.integers(500000, 510000, n)],
        "StockCode":   rng.choice(["85123A", "71053", "84406B", "22752", "21730"], n),
        "Description": rng.choice(["CREAM HANGING HEART", "WHITE METAL LANTERN"], n),
        "Quantity":    rng.integers(1, 20, n),
        "InvoiceDate": pd.date_range("2011-01-01", periods=n, freq="1h"),
        "UnitPrice":   rng.uniform(0.5, 20.0, n),
        "CustomerID":  rng.integers(12000, 12050, n).astype(float),
        "Country":     rng.choice(["United Kingdom", "Germany"], n),
    })


@pytest.fixture
def clean_df(raw_retail_df):
    return clean(raw_retail_df)


@pytest.fixture
def treatment_df(clean_df):
    feats = build_customer_features(clean_df)
    return build_treatment_dataset(clean_df, feats, sample_customers=5000)


# ── Cleaning Tests ────────────────────────────────────────────────────────────

class TestCleaning:

    def test_drops_null_customerid(self, raw_retail_df):
        raw_retail_df.loc[0, "CustomerID"] = np.nan
        df = clean(raw_retail_df)
        assert df["customerid"].isna().sum() == 0

    def test_drops_cancellations(self, raw_retail_df):
        raw_retail_df.loc[0, "InvoiceNo"] = "C500001"
        df = clean(raw_retail_df)
        assert not df["invoiceno"].astype(str).str.startswith("C").any()

    def test_positive_quantity_and_price(self, clean_df):
        assert (clean_df["quantity"] > 0).all()
        assert (clean_df["unitprice"] > 0).all()

    def test_revenue_column_exists(self, clean_df):
        assert "revenue" in clean_df.columns
        assert (clean_df["revenue"] > 0).all()

    def test_lowercase_columns(self, clean_df):
        for col in clean_df.columns:
            assert col == col.lower(), f"Column not lowercase: {col}"

    def test_invoicedate_is_datetime(self, clean_df):
        assert pd.api.types.is_datetime64_any_dtype(clean_df["invoicedate"])


# ── Feature Engineering Tests ─────────────────────────────────────────────────

class TestFeatureEngineering:

    def test_user_product_matrix_columns(self, clean_df):
        upm = build_user_product_matrix(clean_df)
        assert set(upm.columns) == {"customerid", "stockcode", "purchase_count"}

    def test_user_product_counts_positive(self, clean_df):
        upm = build_user_product_matrix(clean_df)
        assert (upm["purchase_count"] > 0).all()

    def test_customer_features_shape(self, clean_df):
        feats = build_customer_features(clean_df)
        assert len(feats) == clean_df["customerid"].nunique()
        for col in FEATURE_COLS:
            assert col in feats.columns, f"Missing feature: {col}"

    def test_customer_features_no_nulls(self, clean_df):
        feats = build_customer_features(clean_df)
        assert feats[FEATURE_COLS].isna().sum().sum() == 0

    def test_reorder_ratio_bounded(self, clean_df):
        feats = build_customer_features(clean_df)
        assert feats["reorder_ratio"].between(0, 1).all()

    def test_product_diversity_bounded(self, clean_df):
        feats = build_customer_features(clean_df)
        assert feats["product_diversity"].between(0, 1).all()


# ── Treatment Dataset Tests ───────────────────────────────────────────────────

class TestTreatmentDataset:

    def test_treatment_column_binary(self, treatment_df):
        assert set(treatment_df["treatment"].unique()).issubset({0, 1})

    def test_converted_column_binary(self, treatment_df):
        assert set(treatment_df["converted"].unique()).issubset({0, 1})

    def test_treatment_rate_approx_half(self, treatment_df):
        rate = treatment_df["treatment"].mean()
        assert 0.4 < rate < 0.6, f"Treatment rate {rate:.2%} far from 50%"

    def test_no_nulls_in_features(self, treatment_df):
        assert treatment_df[FEATURE_COLS].isna().sum().sum() == 0


# ── Uplift Model Tests ────────────────────────────────────────────────────────

class TestUpliftModel:

    def test_split_balance(self, treatment_df):
        train, test = split_dataset(treatment_df)
        assert abs(train["treatment"].mean() - test["treatment"].mean()) < 0.05

    def test_fit_predict_shape(self, treatment_df):
        train, test = split_dataset(treatment_df)
        suite = UpliftModelSuite(base_estimator="xgboost")
        suite.fit(train)
        preds = suite.predict_uplift(test)
        assert len(preds) == len(test)
        for col in ["uplift_x_learner", "uplift_t_learner",
                    "uplift_s_learner", "uplift_ensemble"]:
            assert col in preds.columns

    def test_predictions_finite(self, treatment_df):
        train, test = split_dataset(treatment_df)
        suite = UpliftModelSuite()
        suite.fit(train)
        preds = suite.predict_uplift(test)
        for col in ["uplift_x_learner", "uplift_t_learner", "uplift_s_learner"]:
            assert np.isfinite(preds[col]).all(), f"Non-finite values in {col}"

    def test_ensemble_is_mean(self, treatment_df):
        train, test = split_dataset(treatment_df)
        suite = UpliftModelSuite()
        suite.fit(train)
        preds = suite.predict_uplift(test)
        expected = (
            preds["uplift_x_learner"]
            + preds["uplift_t_learner"]
            + preds["uplift_s_learner"]
        ) / 3
        np.testing.assert_allclose(preds["uplift_ensemble"], expected, rtol=1e-5)

    def test_unfitted_raises(self, treatment_df):
        suite = UpliftModelSuite()
        with pytest.raises(RuntimeError):
            suite.predict_uplift(treatment_df)

    def test_save_load_roundtrip(self, treatment_df, tmp_path):
        train, test = split_dataset(treatment_df)
        suite = UpliftModelSuite()
        suite.fit(train)
        path = str(tmp_path / "suite.pkl")
        suite.save(path)
        loaded = UpliftModelSuite.load(path)
        pd.testing.assert_frame_equal(
            suite.predict_uplift(test).reset_index(drop=True),
            loaded.predict_uplift(test).reset_index(drop=True),
        )


# ── Integration Test ──────────────────────────────────────────────────────────

class TestIntegration:

    def test_full_pipeline_smoke(self, clean_df):
        """End-to-end: data → features → treatment → uplift → top-K."""
        feats = build_customer_features(clean_df)
        t_df  = build_treatment_dataset(clean_df, feats, sample_customers=5000)
        train, test = split_dataset(t_df)

        suite = UpliftModelSuite()
        suite.fit(train)
        preds = suite.predict_uplift(test)

        n_target = max(1, int(len(test) * 0.2))
        combined = pd.concat(
            [test.reset_index(drop=True), preds.reset_index(drop=True)], axis=1
        )
        top = combined.nlargest(n_target, "uplift_ensemble")
        assert len(top) == n_target
        assert "converted" in top.columns
