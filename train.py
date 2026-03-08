"""
End-to-end training pipeline for the Post-Purchase Cross-Sell Recommender.

Uses UCI Online Retail dataset (id=352) — no login required, ~22MB.

Run:
    python train.py
    python train.py --help
    python train.py --epochs 10 --dims 32 --estimator lightgbm
"""

import argparse
import logging
import os
import sys

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import (
    load_online_retail,
    clean,
    build_user_product_matrix,
    build_customer_features,
    build_treatment_dataset,
    FEATURE_COLS,
)
from src.models.embedding_model import CoPurchaseEmbeddingModel
from src.models.uplift_model import UpliftModelSuite, split_dataset
from src.evaluation.plots import (
    plot_uplift_curves,
    plot_targeting_policy,
    plot_uplift_distributions,
    plot_feature_importance,
    plot_revenue_lift,
    print_metrics_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train the post-purchase cross-sell system.")
    p.add_argument("--output-dir",  default="results",   help="Output directory")
    p.add_argument("--epochs",      type=int, default=15, help="LightFM epochs")
    p.add_argument("--dims",        type=int, default=32, help="Embedding dimensions")
    p.add_argument("--estimator",   choices=["xgboost", "lightgbm"], default="xgboost")
    p.add_argument("--sample-customers", type=int, default=30_000)
    p.add_argument("--skip-embeddings",  action="store_true",
                   help="Load cached embedding model if available")
    p.add_argument("--experiment-name",  default="post_purchase_crosssell")
    return p.parse_args()


def main():
    args = parse_args()
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name="full_pipeline"):
        mlflow.log_params(vars(args))

        # ── 1. Load & Clean ───────────────────────────────────────────────────
        logger.info("Step 1/5 — Loading UCI Online Retail dataset...")
        raw = load_online_retail()
        df = clean(raw)
        mlflow.log_metrics({
            "n_transactions": len(df),
            "n_customers":    df["customerid"].nunique(),
            "n_products":     df["stockcode"].nunique(),
        })

        # ── 2. Train Co-Purchase Embedding Model ──────────────────────────────
        emb_path = "models/embedding_model.pkl"
        if args.skip_embeddings and os.path.exists(emb_path):
            logger.info("Step 2/5 — Loading cached embedding model...")
            emb_model = CoPurchaseEmbeddingModel.load(emb_path)
        else:
            logger.info("Step 2/5 — Training LightFM co-purchase embeddings...")
            upm = build_user_product_matrix(df)
            emb_model = CoPurchaseEmbeddingModel(
                n_components=args.dims,
                epochs=args.epochs,
            )
            emb_model.fit(upm)
            emb_model.save(emb_path)
            mlflow.log_metric("embedding_dims", args.dims)

        # ── 3. Build Features & Treatment Dataset ─────────────────────────────
        logger.info("Step 3/5 — Building customer features and treatment dataset...")
        customer_feats = build_customer_features(df)
        treatment_df = build_treatment_dataset(
            df, customer_feats, sample_customers=args.sample_customers
        )

        train_df, test_df = split_dataset(treatment_df)
        mlflow.log_metrics({
            "n_train": len(train_df),
            "n_test":  len(test_df),
            "train_conversion_rate": train_df["converted"].mean(),
            "test_conversion_rate":  test_df["converted"].mean(),
        })
        logger.info(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

        # ── 4. Train Uplift Models ────────────────────────────────────────────
        logger.info("Step 4/5 — Training uplift model suite...")
        suite = UpliftModelSuite(base_estimator=args.estimator)
        suite.fit(train_df, feature_cols=FEATURE_COLS)
        suite.save("models/uplift_suite.pkl")

        # ── 5. Evaluate & Plot ────────────────────────────────────────────────
        logger.info("Step 5/5 — Evaluating and generating plots...")
        uplift_preds = suite.predict_uplift(test_df)
        metrics = suite.evaluate(test_df, log_mlflow=False)
        mlflow.log_metrics(metrics)
        print_metrics_table(metrics)

        # Save scored test set
        results = pd.concat(
            [test_df.reset_index(drop=True), uplift_preds.reset_index(drop=True)],
            axis=1,
        ).loc[:, lambda d: ~d.columns.duplicated()]
        results.to_csv(os.path.join(args.output_dir, "test_uplift_scores.csv"), index=False)

        # Plots
        plot_uplift_curves(
            test_df, uplift_preds,
            save_path=os.path.join(plots_dir, "uplift_curves.png"),
        )
        plot_targeting_policy(
            test_df, uplift_preds,
            save_path=os.path.join(plots_dir, "policy_comparison.png"),
        )
        plot_uplift_distributions(
            uplift_preds,
            save_path=os.path.join(plots_dir, "uplift_distributions.png"),
        )
        plot_feature_importance(
            test_df, uplift_preds, FEATURE_COLS,
            save_path=os.path.join(plots_dir, "feature_importance.png"),
        )
        plot_revenue_lift(
            test_df, uplift_preds,
            save_path=os.path.join(plots_dir, "revenue_lift.png"),
        )

        mlflow.log_artifacts(plots_dir, artifact_path="plots")
        mlflow.log_artifact("models/uplift_suite.pkl", artifact_path="models")

        logger.info(f"\n✅ Done.  Results → {args.output_dir}/")
        logger.info("   MLflow UI: mlflow ui --port 5000")


if __name__ == "__main__":
    main()
