import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from data_analysis.qual import QualitativeAnalysis


class TestQualitativeAnalysis(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({"text": ["this is a positive statement",
                                         "this is a neutral statement",
                                         "this is a negative statement"]})
        self.qual_analyzer = QualitativeAnalysis(self.df)

    def test_get_qualitative_data(self):
        result_df = self.qual_analyzer.get_qualitative_data()
        self.assertIsNotNone(result_df)


if __name__ == "__main__":
    unittest.main()
