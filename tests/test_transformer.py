import unittest
import pandas as pd
import numpy as np
from data_transformation.transformer import DataTransformer


class TestTransformer(unittest.TestCase):

    def setUp(self):
        self.data = {'A': [1, 2, 3], 'B': [1.5, 2.5, 3.5], 'C': ['a', 'b', 'c']}
        self.df = pd.DataFrame(self.data)

    def test_one_hot_encode(self):
        transformer = DataTransformer(self.df)
        transformed = transformer.one_hot_encode('C').get_transformed_data()
        print(transformed)
        self.assertTrue('C_1' in transformed.columns)

    def test_log_transform(self):
        transformer = DataTransformer(self.df)
        transformed = transformer.log_transform('B').get_transformed_data()
        expected_vals = np.log1p(self.df['B'])
        self.assertTrue(np.allclose(transformed['B'], expected_vals))

    def test_apply_custom_transform(self):
        square_func = lambda x: x ** 2
        transformer = DataTransformer(self.df)
        transformed = transformer.apply_custom_transform('A', square_func).get_transformed_data()
        self.assertTrue(np.array_equal(transformed['A'], np.array([1, 4, 9])))

    def test_bin_data(self):
        transformer = DataTransformer(self.df)
        transformed = transformer.bin_data('B', bins=2, labels=['Low', 'High']).get_transformed_data()
        expected_vals = pd.cut(self.df['B'], bins=2, labels=['Low', 'High'])
        self.assertTrue(np.array_equal(transformed['B'], expected_vals))


if __name__ == '__main__':
    unittest.main()
