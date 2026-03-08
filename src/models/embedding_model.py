"""
Co-purchase embedding model using LightFM (implicit feedback collaborative filtering).

Learns product and user embeddings from basket co-occurrence in the
UCI Online Retail dataset. Product embeddings capture semantic product
similarity — items bought together have similar embeddings.

These embeddings serve two roles:
  1. Direct cross-sell recommendation (k-nearest products by embedding similarity)
  2. Feature input to the uplift model (user embedding as behavioural signal)
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
import os

logger = logging.getLogger(__name__)


class CoPurchaseEmbeddingModel:
    """
    Trains product and user embeddings via LightFM WARP loss on purchase data.

    WARP (Weighted Approximate-Rank Pairwise) loss is optimised for
    top-K recommendation tasks — it directly maximises the rank of
    observed purchases over unobserved ones.
    """

    def __init__(
        self,
        n_components: int = 32,
        loss: str = "warp",
        epochs: int = 15,
        num_threads: int = 2,
    ):
        self.n_components = n_components
        self.loss = loss
        self.epochs = epochs
        self.num_threads = num_threads

        self.model = LightFM(no_components=n_components, loss=loss, random_state=42)
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.interaction_matrix: csr_matrix = None
        self.is_fitted = False

    def fit(self, user_product_matrix: pd.DataFrame) -> "CoPurchaseEmbeddingModel":
        """
        Train on a DataFrame with columns:
            customerid, stockcode, purchase_count
        """
        logger.info(
            f"Fitting LightFM ({self.loss} loss, {self.n_components} dims, "
            f"{self.epochs} epochs)..."
        )
        df = user_product_matrix.copy()
        df["user_idx"] = self.user_encoder.fit_transform(df["customerid"])
        df["item_idx"] = self.item_encoder.fit_transform(df["stockcode"])

        n_users = df["user_idx"].max() + 1
        n_items = df["item_idx"].max() + 1

        # Log(1 + count) as confidence weight — standard for implicit feedback
        weights = np.log1p(df["purchase_count"].values).astype(np.float32)

        self.interaction_matrix = csr_matrix(
            (weights, (df["user_idx"].values, df["item_idx"].values)),
            shape=(n_users, n_items),
        )

        self.model.fit(
            self.interaction_matrix,
            epochs=self.epochs,
            num_threads=self.num_threads,
            verbose=True,
        )
        self.is_fitted = True
        logger.info("LightFM training complete.")
        return self

    def get_item_embeddings(self) -> np.ndarray:
        """Return product embedding matrix [n_items × n_components]."""
        self._check_fitted()
        return self.model.item_embeddings

    def get_user_embeddings(self) -> np.ndarray:
        """Return user embedding matrix [n_users × n_components]."""
        self._check_fitted()
        return self.model.user_embeddings

    def recommend(
        self,
        customer_id: int,
        purchased_stockcodes: list,
        top_k: int = 10,
        exclude_purchased: bool = True,
    ) -> pd.DataFrame:
        """
        Recommend cross-sell products for a given customer.

        Returns DataFrame [stockcode, score] sorted by score descending.
        Falls back to item popularity for cold-start customers.
        """
        self._check_fitted()

        if customer_id not in self.user_encoder.classes_:
            logger.warning(f"Customer {customer_id} unseen — using popularity fallback.")
            return self._popularity_fallback(purchased_stockcodes, top_k)

        user_idx = self.user_encoder.transform([customer_id])[0]
        n_items = len(self.item_encoder.classes_)
        scores = self.model.predict(
            user_idx, np.arange(n_items), num_threads=self.num_threads
        )

        recs = (
            pd.DataFrame({"stockcode": self.item_encoder.classes_, "score": scores})
            .sort_values("score", ascending=False)
        )
        if exclude_purchased:
            recs = recs[~recs["stockcode"].isin(purchased_stockcodes)]

        return recs.head(top_k).reset_index(drop=True)

    def _popularity_fallback(self, exclude_ids: list, top_k: int) -> pd.DataFrame:
        item_counts = np.asarray(self.interaction_matrix.sum(axis=0)).flatten()
        top_idx = np.argsort(item_counts)[::-1]
        top_items = self.item_encoder.classes_[top_idx]
        fb = pd.DataFrame({"stockcode": top_items, "score": item_counts[top_idx]})
        return fb[~fb["stockcode"].isin(exclude_ids)].head(top_k).reset_index(drop=True)

    def evaluate(self, test_interactions: csr_matrix, k: int = 10) -> dict:
        """Precision@K and Recall@K on held-out interactions."""
        self._check_fitted()
        p = precision_at_k(
            self.model, test_interactions, k=k, num_threads=self.num_threads
        ).mean()
        r = recall_at_k(
            self.model, test_interactions, k=k, num_threads=self.num_threads
        ).mean()
        return {"precision@k": float(p), "recall@k": float(r), "k": k}

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before use.")

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "CoPurchaseEmbeddingModel":
        m = joblib.load(path)
        logger.info(f"Model loaded ← {path}")
        return m
