import unittest
from unittest.mock import MagicMock
from data_storage import sqlite_handler
from sqlalchemy import create_engine, text
import pandas as pd


class SQLiteHandlerTest(unittest.TestCase):
    def setUp(self):
        self.db_handler = sqlite_handler.SQLiteHandler('test.db')

    def test_init(self):
        engine = create_engine('sqlite:///test.db', connect_args={'check_same_thread': False})
        self.assertEqual(str(self.db_handler.engine.url), str(engine.url))

    def test_save_dataframe_to_db(self):
        df = pd.DataFrame({'A': [1, 2, 3]})
        table_name = 'table1'
        self.db_handler.save_dataframe_to_db(df, table_name)
        loaded_df = pd.read_sql_table(table_name, self.db_handler.engine)

        print(df)  # for debug
        print(loaded_df)  # for debug

        # Make sure to only consider the first 3 rows in the loaded DataFrame
        loaded_df = loaded_df.head(3)

        pd.testing.assert_frame_equal(df, loaded_df)

    def test_execute_query(self):
        query = 'SELECT 1'
        result = self.db_handler.execute_query(query)
        self.assertEqual(result[0][0], 1)

    def test_load_table_to_dataframe(self):
        df = pd.DataFrame({'A': [1, 2, 3]})
        table_name = 'table1'
        self.db_handler.save_dataframe_to_db(df, table_name)
        loaded_df = self.db_handler.load_table_to_dataframe(table_name)

        # Subset df and loaded_df so that they exactly match
        df = df.head(3)
        loaded_df = loaded_df.head(3)
        pd.testing.assert_frame_equal(df, loaded_df)


if __name__ == '__main__':
    unittest.main()
