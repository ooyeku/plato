from sqlalchemy import create_engine, text
import pandas as pd
from utils.logger import logger


class SQLiteHandler:
    """
    SQLiteHandler class is used to handle database operations in SQLite.

    Methods:
    - __init__(self, db_name='plato.db'):
        Constructor method that initializes the SQLiteHandler object with a database name.
        :param db_name: Optional parameter specifying the name of the database (default is 'plato.db').
        :return: None

    - create_connection(self):
        Creates a connection to the SQLite database.
        :return: A connection object.

    - close_connection(self):
        Closes the connection to the SQLite database.
        :return: None

    - save_dataframe_to_db(self, df, table_name):
        Saves a pandas DataFrame to a table in the SQLite database.
        :param df: The DataFrame to be saved.
        :param table_name: The name of the table to save the DataFrame.
        :return: None

    - execute_query(self, query):
        Executes a SQL query on the SQLite database.
        :param query: The SQL query to be executed.
        :return: The result of the query as a list of tuples.

    - load_table_to_dataframe(self, table_name):
        Loads a table from the SQLite database into a pandas DataFrame.
        :param table_name: The name of the table to load.
        :return: The loaded pandas DataFrame, or None if an error occurs.
    """
    def __init__(self, db_name='plato.db'):
        self.engine = create_engine(f'sqlite:///{db_name}')
        logger.info(f"Database connection created with {db_name}")

    def create_connection(self):
        return self.engine.connect()

    def close_connection(self):
        self.engine.dispose()
        logger.info(f"Database connection closed")

    def save_dataframe_to_db(self, df, table_name):
        try:
            df.to_sql(table_name, con=self.engine, if_exists='replace', index=False)
            logger.info(f"DataFrame saved to table {table_name}")
        except Exception as e:
            logger.error(f"Error saving DataFrame to table {table_name}: {e}")

    def execute_query(self, query):
        try:
            with self.create_connection() as conn:
                result = conn.execute(text(query))
                logger.info(f"Query executed: {query}")
                return result.fetchall()
        except Exception as e:
            logger.error(f"Error executing query: {query}: {e}")

    def load_table_to_dataframe(self, table_name):
        try:
            df = pd.read_sql_table(table_name, con=self.engine)
            logger.info(f"Table {table_name} loaded into DataFrame")
            return df
        except Exception as e:
            logger.error(f"Error loading table {table_name} into DataFrame: {e}")
            return None
