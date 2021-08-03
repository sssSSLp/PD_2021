import unittest

import numpy as np
import pandas as pd

from pdc.preprocessing.rank_converter import RankConverter


class TestConvertToRank(unittest.TestCase):

    def test_transform(self):
        read_count_matrix = np.asarray([[0, 1, 2, ],
                                        [3, 4, 5, ],
                                        [6, 7, 8, ],
                                        [9, 10, 11], ])
        read_count_df = pd.DataFrame(read_count_matrix, columns=['gene0', 'gene1', 'gene2'],
                                     index=['sample0', 'sample1', 'sample2', 'sample3'])
        params = {}
        converter = RankConverter(params)
        converter.fit(x=read_count_df)
        actual = converter.transform(x=read_count_df)

        self.assertEqual(actual.index.values.tolist(), ['sample0', 'sample1', 'sample2',
                                                        'sample3'])
        self.assertEqual(actual.columns.values.tolist(), ['gene0', 'gene1', 'gene2'])
        np.testing.assert_almost_equal(actual.values,
                                       np.asarray(
                                           [[1., 2., 3.],
                                            [1., 2., 3.],
                                            [1., 2., 3.],
                                            [1., 2., 3.]]),
                                       decimal=5)

    def test_transform_with_normalizing(self):
        read_count_matrix = np.asarray([[0, 1, 2, ],
                                        [3, 4, 5, ],
                                        [6, 7, 8, ],
                                        [9, 10, 11], ])
        read_count_df = pd.DataFrame(read_count_matrix, columns=['gene0', 'gene1', 'gene2'],
                                     index=['sample0', 'sample1', 'sample2', 'sample3'])
        params = {"normalizing": True}
        converter = RankConverter(params)
        converter.fit(x=read_count_df)
        actual = converter.transform(x=read_count_df)

        self.assertEqual(actual.index.values.tolist(), ['sample0', 'sample1', 'sample2',
                                                        'sample3'])
        self.assertEqual(actual.columns.values.tolist(), ['gene0', 'gene1', 'gene2'])
        np.testing.assert_almost_equal(actual.values,
                                       np.asarray(
                                           [[0.33333, 0.66667, 1.],
                                            [0.33333, 0.66667, 1.],
                                            [0.33333, 0.66667, 1.],
                                            [0.33333, 0.66667, 1.]]),
                                       decimal=5)
