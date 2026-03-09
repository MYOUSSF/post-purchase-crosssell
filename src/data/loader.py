"""
Data loading and preprocessing for the UCI Online Retail dataset (id=352).

Dataset source: https://archive.ics.uci.edu/dataset/352/online+retail
  - 541,909 rows, 8 columns
  - UK-based online gift retailer, Dec 2010 – Dec 2011
  - Columns: InvoiceNo, StockCode, Description, Quantity,
             InvoiceDate, UnitPrice, CustomerID, Country

Fetch without login:
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=352)

Schema mapping to project concepts:
    InvoiceNo  → order / basket identifier
    StockCode  → product_id
    CustomerID → user_id
    Quantity   → units purchased
    UnitPrice  → item price
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CACHE_PATH = "data/online_retail.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_online_retail(cache_path: str = CACHE_PATH) -> pd.DataFrame:
    """
    Load UCI Online Retail dataset.
    Caches locally after first download to avoid repeated network calls.
    """
    if os.path.exists(cache_path):
        logger.info(f"Loading from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info("Downloading UCI Online Retail dataset (id=352, ~22MB)...")
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=352)
        df = dataset.data.features
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch dataset: {e}\n"
            "Ensure internet access and ucimlrepo installed:\n"
            "  pip install ucimlrepo"
        )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info(f"Cached to {cache_path} ({len(df):,} rows)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Online Retail data.

    Steps:
      - Drop rows without CustomerID (anonymous sessions — no user history)
      - Drop cancelled invoices (InvoiceNo starting with 'C')
      - Remove non-positive Quantity and UnitPrice
      - Parse InvoiceDate to datetime
      - Standardise column names to snake_case
    """
    df = df.copy()

    # Standardise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Drop anonymous rows
    before = len(df)
    df = df.dropna(subset=["customerid"])
    logger.info(f"Dropped {before - len(df):,} rows without CustomerID")

    # Drop cancellations
    df = df[~df["invoiceno"].astype(str).str.startswith("C")]

    # Keep only valid transactions
    df = df[(df["quantity"] > 0) & (df["unitprice"] > 0)]

    # Parse date
    df["invoicedate"] = pd.to_datetime(df["invoicedate"])

    # Add revenue column
    df["revenue"] = df["quantity"] * df["unitprice"]

    # Clean customerid to int
    df["customerid"] = df["customerid"].astype(int)

    # Normalise stockcode — some are non-product admin codes
    df = df[df["stockcode"].astype(str).str.match(r"^\d{5}[A-Z]*$")]

    logger.info(
        f"Clean dataset: {len(df):,} rows | "
        f"{df['customerid'].nunique():,} customers | "
        f"{df['stockcode'].nunique():,} products | "
        f"{df['invoiceno'].nunique():,} invoices"
    )
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def build_user_product_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate (customer, product) purchase counts.
    Used to train the LightFM co-purchase embedding model.
    """
    matrix = (
        df.groupby(["customerid", "stockcode"])
        .agg(purchase_count=("quantity", "sum"))
        .reset_index()
    )
    logger.info(f"User-product matrix: {len(matrix):,} interactions")
    return matrix


def build_co_purchase_pairs(df: pd.DataFrame, min_support: int = 20) -> pd.DataFrame:
    """
    Find products frequently bought together in the same invoice (basket).
    Used for understanding product affinity structure.
    """
    logger.info("Building co-purchase pairs...")
    basket = df[["invoiceno", "stockcode"]].drop_duplicates()
    pairs = basket.merge(basket, on="invoiceno", suffixes=("_a", "_b"))
    pairs = pairs[pairs["stockcode_a"] < pairs["stockcode_b"]]
    pair_counts = (
        pairs.groupby(["stockcode_a", "stockcode_b"])
        .size()
        .reset_index(name="co_purchase_count")
    )
    pair_counts = pair_counts[pair_counts["co_purchase_count"] >= min_support]
    logger.info(f"Found {len(pair_counts):,} product pairs with support ≥ {min_support}")
    return pair_counts


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM-style behavioural features per customer.
    These are used as covariates in the uplift model.

    Features:
      total_invoices       — number of distinct orders
      total_products       — number of distinct products purchased
      total_items          — total units bought
      total_revenue        — total spend
      avg_order_value      — mean revenue per order
      unique_countries     — 1 if customer buys from multiple countries (rare)
      avg_days_between_ord — mean inter-order gap (recency proxy)
      product_diversity    — unique products / total items (breadth of taste)
      reorder_ratio        — products bought more than once / total products
    """
    snapshot_date = df["invoicedate"].max()

    agg = (
        df.groupby("customerid")
        .agg(
            total_invoices=("invoiceno", "nunique"),
            total_products=("stockcode", "nunique"),
            total_items=("quantity", "sum"),
            total_revenue=("revenue", "sum"),
            first_purchase=("invoicedate", "min"),
            last_purchase=("invoicedate", "max"),
        )
        .reset_index()
    )

    agg["avg_order_value"] = agg["total_revenue"] / agg["total_invoices"]
    agg["days_active"] = (agg["last_purchase"] - agg["first_purchase"]).dt.days + 1
    agg["avg_days_between_ord"] = agg["days_active"] / agg["total_invoices"].clip(lower=1)
    agg["product_diversity"] = agg["total_products"] / agg["total_items"].clip(lower=1)
    agg["recency_days"] = (snapshot_date - agg["last_purchase"]).dt.days

    # Reorder ratio: products bought in >1 invoice / total products
    product_invoice_counts = (
        df.groupby(["customerid", "stockcode"])["invoiceno"]
        .nunique()
        .reset_index(name="n_invoices")
    )
    reorder = (
        product_invoice_counts[product_invoice_counts["n_invoices"] > 1]
        .groupby("customerid")
        .size()
        .reset_index(name="reordered_products")
    )
    agg = agg.merge(reorder, on="customerid", how="left")
    agg["reordered_products"] = agg["reordered_products"].fillna(0)
    agg["reorder_ratio"] = agg["reordered_products"] / agg["total_products"].clip(lower=1)

    agg = agg.drop(columns=["first_purchase", "last_purchase", "days_active", "reordered_products"])
    logger.info(f"Customer features: {len(agg):,} customers × {agg.shape[1]} features")
    return agg


def build_treatment_dataset(
    df: pd.DataFrame,
    customer_features: pd.DataFrame,
    sample_customers: int = 30_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build the dataset used for uplift modelling.

    Treatment definition:
      - treatment=1: customer received a post-purchase cross-sell recommendation
        (simulated via deterministic hash split — mirrors hash-based A/B bucketing)
      - treatment=0: customer received no recommendation

    Outcome (converted=1):
      - Customer purchased a product from a different category than their
        most-purchased category (i.e., cross-category purchase occurred)

    Note: Treatment is simulated from observational data as is standard when
    an actual A/B log is unavailable. Clearly documented throughout.
    """
    rng = np.random.default_rng(random_state)

    customers = customer_features["customerid"].unique()
    if len(customers) > sample_customers:
        customers = rng.choice(customers, size=sample_customers, replace=False)

    feats = customer_features[customer_features["customerid"].isin(customers)].copy()

    # Simulated treatment: hash-based deterministic split
    feats["treatment"] = (feats["customerid"] % 2 == 0).astype(int)

    # Outcome: did the customer buy from a non-primary stockcode prefix?
    # We use first 2 chars of stockcode as a proxy category
    df_sub = df[df["customerid"].isin(customers)].copy()
    df_sub["category"] = df_sub["stockcode"].astype(str).str[:2]

    primary_cat = (
        df_sub.groupby(["customerid", "category"])["quantity"]
        .sum()
        .reset_index()
        .sort_values("quantity", ascending=False)
        .drop_duplicates("customerid")
        .rename(columns={"category": "primary_category"})
    )
    df_sub = df_sub.merge(primary_cat[["customerid", "primary_category"]], on="customerid", how="left")
    df_sub["cross_cat"] = (df_sub["category"] != df_sub["primary_category"]).astype(int)

    cross_cat_flag = (
        df_sub.groupby("customerid")["cross_cat"]
        .max()
        .reset_index()
        .rename(columns={"cross_cat": "converted"})
    )

    result = feats.merge(cross_cat_flag, on="customerid", how="left")
    result["converted"] = result["converted"].fillna(0).astype(int)

    logger.info(
        f"Treatment dataset: {len(result):,} customers | "
        f"Treatment rate: {result['treatment'].mean():.2%} | "
        f"Conversion rate: {result['converted'].mean():.2%}"
    )
    return result


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
