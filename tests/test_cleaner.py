import unittest
import pandas as pd
import numpy as np
from data_transformation.cleaner import DataCleaner


class TestDataCleaner(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'A': ['foo', 'foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'one', 'two', 'two', 'one'],
            'C': [1.0, np.NaN, 2.5, -3.0, 2.3],
            'D': [1, 1, 2, 3, 5]
        })
        self.cleaner = DataCleaner(self.data)

    def test_remove_duplicates(self):
        self.data = pd.DataFrame({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'two', 'two', 'one'],
            'C': [1.0, 2.5, -3.0, 2.3],
            'D': [1, 2, 3, 5]
        })
        self.cleaner = DataCleaner(self.data)
        self.cleaner.remove_duplicates()
        cleaned_data = self.cleaner.get_cleaned_data()
        self.assertEqual(len(cleaned_data), 4)

    def test_fill_missing_values(self):
        self.cleaner.fill_missing_values(strategy='mean')
        cleaned_data = self.cleaner.get_cleaned_data()
        self.assertFalse(cleaned_data.isnull().values.any())

    def test_drop_missing_values(self):
        self.cleaner.drop_missing_values()
        cleaned_data = self.cleaner.get_cleaned_data()
        self.assertFalse(cleaned_data.isnull().values.any())

    def test_replace_values(self):
        self.cleaner.replace_values('foo', 'spam')
        cleaned_data = self.cleaner.get_cleaned_data()
        self.assertEqual(cleaned_data['A'].tolist(), ['spam', 'spam', 'spam', 'bar', 'bar'])

    def test_convert_data_types(self):
        self.cleaner.convert_data_types(['C'], float)
        cleaned_data = self.cleaner.get_cleaned_data()
        self.assertTrue(pd.api.types.is_float_dtype(cleaned_data['C']))

    def test_normalize_data(self):
        self.cleaner.normalize_data(['D'])
        cleaned_data = self.cleaner.get_cleaned_data()
        self.assertEqual(cleaned_data['D'].max(), 1.0)
        self.assertEqual(cleaned_data['D'].min(), 0.0)

    def test_standardize_data(self):
        self.cleaner.standardize_data(['D'])
        cleaned_data = self.cleaner.get_cleaned_data()
        self.assertAlmostEqual(cleaned_data['D'].mean(), 0.0, places=8)  # 0.0 mean
        self.assertAlmostEqual(cleaned_data['D'].std(), 1.0, places=8)  # 1.0 std dev


if __name__ == "__main__":
    unittest.main()
