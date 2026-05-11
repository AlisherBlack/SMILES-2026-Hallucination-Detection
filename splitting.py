"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


N_SPLITS_DEFAULT = 5
VAL_SIZE_DEFAULT = 0.15
RANDOM_STATE_DEFAULT = 42


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    n_splits: int = N_SPLITS_DEFAULT,
    val_size: float = VAL_SIZE_DEFAULT,
    random_state: int = RANDOM_STATE_DEFAULT,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Stratified K-fold split with an inner train/val partition.

    For each of ``n_splits`` outer folds the indices are partitioned into
    ``(train, val, test)``: ``test`` is the held-out fold from
    ``StratifiedKFold``; the remaining indices are split into ``train`` and
    ``val`` with a stratified ``val_size`` fraction.

    EDA confirms ``unique_prompts_train == n_train`` and no shared prompts
    between train and test, so a stratified random split is leakage-safe; no
    group-aware split is needed.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
        df:           Optional full DataFrame (unused; kept for API contract).
        n_splits:     Number of outer folds for cross-validation.
        val_size:     Fraction of the (train+val) portion reserved for val.
        random_state: Random seed for reproducible splits.

    Returns:
        A list of ``n_splits`` ``(idx_train, idx_val, idx_test)`` tuples.
    """
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for idx_train_val, idx_test in skf.split(np.zeros(len(y)), y):
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=val_size,
            random_state=random_state,
            stratify=y[idx_train_val],
        )
        splits.append((idx_train, idx_val, idx_test))
    return splits
