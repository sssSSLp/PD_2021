import unittest

import numpy as np
import pandas as pd

from pdc.preprocessing.read_count_feature_selector import ReadCountFeatureSelector


class TestReadCountFeatureSelector(unittest.TestCase):
    def setUp(self):
        x = np.asarray([[0, 1, 2], [3, 4, 5], [6, 0, 8], [9, 0, 11]])

        self.x_df = pd.DataFrame(
            x,
            columns=["gene0", "gene1", "gene2"],
            index=["sample0", "sample1", "sample2", "sample3"],
        )

    def test_transform(self):
        np.random.seed(0)
        params = {"min_read_count": 1, "ratio": 0.8}
        feature_selector = ReadCountFeatureSelector(params)
        feature_selector.fit(self.x_df)
        actual = feature_selector.transform(self.x_df)

        self.assertEqual(
            actual.index.values.tolist(), ["sample0", "sample1", "sample2", "sample3"]
        )
        self.assertEqual(actual.columns.values.tolist(), ["gene2"])
        np.testing.assert_equal(actual.values, np.asarray([[2], [5], [8], [11]]))
        x_transformed = feature_selector.transform(self.x_df)
        indices = feature_selector.get_indices()
        pd.testing.assert_frame_equal(x_transformed, self.x_df.iloc[:, indices])
