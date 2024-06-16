import unittest
from unittest import mock
import pandas as pd
from data_ingestion.crosstab_loader import CrosstabLoader


class TestCrosstabLoader(unittest.TestCase):

    def setUp(self):
        self.crosstab_loader = CrosstabLoader()
        self.file_path = "file_path.xlsx"
        self.sheet_name = "sheet_name"

    @mock.patch('data_ingestion.crosstab_loader.SQLiteHandler')
    @mock.patch('pandas.read_excel')
    def test_load_crosstab(self, mock_read_excel, mock_sqlite_handler):
        mock_df = mock.MagicMock(spec=pd.DataFrame)
        mock_read_excel.return_value = mock_df
        result = self.crosstab_loader.load_crosstab(self.file_path, self.sheet_name)

        self.assertEqual(result, mock_df)
        mock_read_excel.assert_called_once_with(self.file_path, sheet_name=self.sheet_name)


    @mock.patch('data_ingestion.crosstab_loader.SQLiteHandler')
    @mock.patch('pandas.read_excel')
    def test_load_multiple_crosstabs(self, mock_read_excel, mock_sqlite_handler):
        mock_df = mock.MagicMock(spec=pd.DataFrame)
        mock_read_excel.return_value = mock_df

        file_paths = ["file_path1.xlsx", "file_path2.xlsx"]
        sheet_names = ["sheet_name1", "sheet_name2"]

        result = self.crosstab_loader.load_multiple_crosstabs(file_paths, sheet_names)

        expected_result = [mock_df, mock_df]
        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
