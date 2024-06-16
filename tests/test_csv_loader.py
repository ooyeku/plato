import unittest
from unittest.mock import Mock, patch
from data_ingestion import csv_loader


class TestCSVLoader(unittest.TestCase):
    @patch('data_ingestion.csv_loader.ThreadPoolExecutor')
    @patch('data_ingestion.csv_loader.SQLiteHandler')
    def test_init(self, mock_sqlite_handler, mock_executor):
        loader = csv_loader.CSVLoader()
        mock_sqlite_handler.assert_called_once_with('plato.db')
        self.assertEqual(loader.chunk_size, 50000)

    @patch('pandas.read_csv')
    @patch('data_ingestion.csv_loader.log_info')
    @patch('data_ingestion.csv_loader.log_error')
    @patch('data_ingestion.csv_loader.SQLiteHandler')
    def test_load_csv(self, mock_sqlite_handler, mock_log_error, mock_log_info, mock_read_csv):
        mock_sqlite_handler_instance = Mock()
        mock_sqlite_handler.return_value = mock_sqlite_handler_instance
        loader = csv_loader.CSVLoader()
        mock_df = Mock()
        mock_read_csv.return_value = mock_df
        df = loader.load_csv('test.csv')
        mock_read_csv.assert_called_once_with('test.csv')
        self.assertEqual(df, mock_df)

    @patch('pandas.read_csv')
    @patch('data_ingestion.csv_loader.log_info')
    @patch('data_ingestion.csv_loader.log_error')
    @patch('data_ingestion.csv_loader.SQLiteHandler')
    def test_load_csv_not_save_to_db(self, mock_sqlite_handler, mock_log_error, mock_log_info, mock_read_csv):
        mock_sqlite_handler_instance = Mock()
        mock_sqlite_handler.return_value = mock_sqlite_handler_instance
        loader = csv_loader.CSVLoader()
        mock_df = Mock()
        mock_read_csv.return_value.iterrows.return_value = [('a', 'b'), ('c', 'd')]

        # This is the area where we're calling load_csv method with save_to_db=False
        loader.load_csv('test.csv', save_to_db=False)

        # Assert if save_dataframe_to_db has not been called.
        mock_sqlite_handler_instance.save_dataframe_to_db.assert_not_called()


if __name__ == '__main__':
    unittest.main()
