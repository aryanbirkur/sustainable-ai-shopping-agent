"""
recommendation/collaborative/cf_scorer.py

Type: ML (item-item collaborative filtering via cosine similarity)

Builds a user-item interaction matrix from data/processed/interactions_clean.csv
and scores candidate products for a given user based on similarity to items
that user has already interacted with.

Cold start is handled honestly: if user_id is None, unknown, or has zero
interaction history, this returns None for every candidate rather than
fabricating a plausible-looking score. Callers must treat None as
"no signal available", never as an implicit zero.

Edge case handled explicitly: interaction_value is summed per (user, product)
pair, and the value scale includes a negative "dislike" (-2). A product whose
only interaction was e.g. click(+2) then dislike(-2) nets to exactly 0 for
that user. If that's the product's *only* interacted-with user, its entire
column in the matrix is a zero vector, which has no direction -- cosine
similarity would divide by a zero norm and produce NaN. We detect and zero
out those columns explicitly rather than letting NaN silently leak into scores.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config import settings

logger = logging.getLogger(__name__)


class CollaborativeFilteringScorer:
    """Item-item CF over the (~120 user x ~394 product) interaction matrix.

    Why item-item cosine similarity and not matrix factorization: at this
    scale (1,798 interactions, 120 users, 394 products), the matrix is small
    enough to hold densely in memory, and a full SVD/ALS factorization adds
    hyperparameters (latent dims, regularization, epochs) without a clear
    accuracy benefit over a transparent, explainable item-item approach --
    you can literally inspect "why" a score is high. Worth revisiting once
    the interaction log is 10-100x bigger.
    """

    def __init__(self, interactions_path: Optional[str] = None):
        self.interactions_path = interactions_path or settings.MILESTONE5_INTERACTIONS_CLEAN_PATH
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.item_similarity: Optional[pd.DataFrame] = None
        self._fitted = False

    def fit(self):
        """Build the user-item matrix and item-item similarity matrix."""
        try:
            df = pd.read_csv(self.interactions_path)
        except Exception as e:
            logger.error(f"Could not load interactions from {self.interactions_path}: {e}")
            self.user_item_matrix = pd.DataFrame()
            self.item_similarity = pd.DataFrame()
            self._fitted = True
            return

        # A user can interact with the same product more than once (view, then
        # later purchase). Summing interaction_value lets repeated positive
        # signals compound, and lets a later dislike partially or fully cancel
        # an earlier positive interaction -- an explicit modelling choice.
        pivot = df.pivot_table(
            index="user_id",
            columns="product_id",
            values="interaction_value",
            aggfunc="sum",
            fill_value=0.0,
        )
        self.user_item_matrix = pivot

        if pivot.shape[1] > 1:
            matrix = pivot.T.values.astype(float)  # items x users

            # Zero-norm rows (an item whose net interaction summed to exactly 0
            # across all users) would cause a divide-by-zero in cosine_similarity's
            # internal normalization. Detect them up front and compute similarity
            # only over the non-zero rows, then reinsert zero rows/columns of 0.0
            # similarity for those items (i.e. "no measurable signal", not NaN).
            row_norms = np.linalg.norm(matrix, axis=1)
            nonzero_mask = row_norms > 1e-9
            n_items = matrix.shape[0]

            sim_full = np.zeros((n_items, n_items))
            if nonzero_mask.any():
                sim_nonzero = cosine_similarity(matrix[nonzero_mask])
                idx = np.where(nonzero_mask)[0]
                sim_full[np.ix_(idx, idx)] = sim_nonzero

            n_zero_items = int((~nonzero_mask).sum())
            if n_zero_items:
                logger.warning(
                    f"{n_zero_items} product(s) had a net-zero interaction total "
                    f"(e.g. click then dislike cancelling out) -- treating their "
                    f"item-item similarity as 0 instead of letting it divide-by-zero into NaN."
                )

            self.item_similarity = pd.DataFrame(sim_full, index=pivot.columns, columns=pivot.columns)
        else:
            self.item_similarity = pd.DataFrame()

        self._fitted = True
        logger.info(
            f"CF fitted: {pivot.shape[0]} users x {pivot.shape[1]} products, "
            f"{(pivot.values != 0).sum()} nonzero interactions."
        )

    def score(
        self, user_id: Optional[str], candidate_product_ids: List[str]
    ) -> Dict[str, Optional[float]]:
        """
        Return a 0-1 CF score per candidate product for `user_id`, or None per
        candidate if no CF signal is available (cold start / unknown user / no history).
        """
        if not self._fitted:
            self.fit()

        if (
            user_id is None
            or self.user_item_matrix is None
            or self.user_item_matrix.empty
            or user_id not in self.user_item_matrix.index
        ):
            return {pid: None for pid in candidate_product_ids}

        user_row = self.user_item_matrix.loc[user_id]
        interacted_items = user_row[user_row != 0]

        if interacted_items.empty:
            return {pid: None for pid in candidate_product_ids}

        raw_scores: Dict[str, float] = {}
        for pid in candidate_product_ids:
            if self.item_similarity.empty or pid not in self.item_similarity.columns:
                raw_scores[pid] = 0.0  # never interacted with, in the log at all
                continue
            sims = self.item_similarity.loc[pid, interacted_items.index]
            weights_sum = sims.abs().sum()
            raw_scores[pid] = (
                0.0 if weights_sum == 0
                else float((sims * interacted_items.values).sum() / weights_sum)
            )

        # min-max normalize to [0,1] across this candidate set so it blends
        # cleanly alongside the other 0-1 signals
        values = np.array(list(raw_scores.values()))
        vmin, vmax = values.min(), values.max()
        if vmax - vmin < 1e-9:
            return {pid: 0.5 for pid in raw_scores}  # flat signal -> neutral
        return {pid: (v - vmin) / (vmax - vmin) for pid, v in raw_scores.items()}
