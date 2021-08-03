import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd


def _select_feature_with_read_count(
        x: pd.DataFrame, min_read_count: int = 10, ratio: float = 0.8
) -> Any:
    mask = (x >= min_read_count).astype(int).sum(axis=0) >= len(x) * ratio
    selected_columns = mask.astype(np.int).to_numpy().nonzero()[0]
    return selected_columns


class ReadCountFeatureSelector(object):

    def __init__(self, parameters):
        self.params = {
            "min_read_count": parameters.get("min_read_count", 10),
            "ratio": parameters.get("ratio", 0.8),
        }
        self._is_fitted = False

    def fit(self, x: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "ReadCountFeatureSelector":
        _min_read_count = self.params["min_read_count"]
        _ratio = self.params["ratio"]

        self._selected_column_ids = _select_feature_with_read_count(x, _min_read_count, _ratio)
        if len(self._selected_column_ids) == 0:
            warnings.warn(
                "No features would be selected if we apply "
                "the default procedure using specified parameters. "
                "Therefore, we select all features instead.",
                RuntimeWarning,
            )
            self._selected_column_ids = range(x.shape[1])
        self._is_fitted = True
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("you should call fit() before transform()")
        return x.iloc[:, self._selected_column_ids]

    def get_indices(self) -> Any:
        """Returns the indices of selected features"""
        if not self._is_fitted:
            raise RuntimeError("you should call fit() before get_index()")
        return self._selected_column_ids
