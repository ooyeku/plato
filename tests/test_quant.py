import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from data_analysis.quant import QuantitativeAnalysis


class TestQuantitativeAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = {'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1], 'C': [10, 20, 30, 40, 50]}
        cls.df = pd.DataFrame(data)
        cls.quant_analysis = QuantitativeAnalysis(cls.df)

    def test_descriptive_statistics(self):
        results = self.quant_analysis.descriptive_statistics()
        self.assertEqual(results.shape[1], len(self.df.columns))
        self.assertIn('mean', results.index)
        self.assertIn('std', results.index)

    def test_correlation_matrix(self):
        results = self.quant_analysis.correlation_matrix()
        self.assertIn('A', results.columns)
        self.assertIn('B', results.columns)
        self.assertAlmostEqual(abs(results.loc['A', 'B']), abs(self.df['A'].corr(self.df['B'])), places=7)

    def test_linear_regression(self):
        results = self.quant_analysis.linear_regression('A', ['B', 'C'])
        self.assertIsInstance(results['model'], LinearRegression)
        self.assertIn('predictions', results)
        self.assertIn('mse', results)
        self.assertIn('r2', results)

    def test_hypothesis_testing(self):
        results_ttest = self.quant_analysis.hypothesis_testing('A', 'B', test='t-test')
        self.assertIn('statistic', results_ttest)
        self.assertIn('p_value', results_ttest)

        results_anova = self.quant_analysis.hypothesis_testing('A', 'B', test='anova')
        self.assertIn('statistic', results_anova)
        self.assertIn('p_value', results_anova)

        with self.assertRaises(ValueError):
            self.quant_analysis.hypothesis_testing('A', 'B', test='invalid_test')

    def test_get_quantitative_data(self):
        results = self.quant_analysis.get_quantitative_data()
        self.assertEqual(results.shape, self.df.shape)
        self.assertTrue((results.columns == self.df.columns).all())


if __name__ == '__main__':
    unittest.main()
