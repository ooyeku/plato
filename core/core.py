import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from data_modeling.modeler import Modeler
from utils.config import Config
from data_ingestion.csv_loader import CSVLoader
from data_ingestion.crosstab_loader import CrosstabLoader
from data_transformation.cleaner import DataCleaner
from data_transformation.transformer import DataTransformer
from data_analysis.qual import QualitativeAnalysis
from data_analysis.quant import QuantitativeAnalysis
from data_modeling.visualizer import Visualizer
from generators.generate_data import DataGenerator
from data_storage.sqlite_handler import SQLiteHandler
from data_storage.query_builder import QueryBuilder
from utils.logger import logger

logger.setLevel(logging.INFO)


def load_data(file_path: str, file_type: str = 'csv', **kwargs) -> pd.DataFrame:
    if file_type == 'csv':
        loader = CSVLoader()
        return loader.load_csv(file_path, **kwargs)
    elif file_type == 'xlsx' or file_type == 'xls' or file_type == 'excel':
        loader = CrosstabLoader()
        return loader.load_crosstab(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


class Core:
    def __init__(self, config_file: str = 'config.json'):
        self.config = Config(config_file)
        self.executor = ThreadPoolExecutor()
        self.dataframe = None
        self.db_path = 'data.db'
        self.table_name = 'generated_data'
        self.query_builder = QueryBuilder()

    def generate_data(self, num_rows: int = None, columns: List[Dict[str, Any]] = None,
                      save_path: str = 'generated_data.csv') -> pd.DataFrame:
        """
        This method generates artificial data and saves it in a CSV file. It returns the generated data as a pandas DataFrame.

        :param num_rows: The number of rows to generate in the DataFrame. If not provided, it defaults to the value specified in the configuration.
        :param columns: A list of dictionaries representing the columns to include in the DataFrame. Each dictionary should contain the keys 'name' (column name), 'data_type' (column data type), and optionally 'options' (additional options for the column).
        :param save_path: The file path where the generated data will be saved. If not provided, it defaults to 'generated_data.csv'.

        :return: The generated data as a pandas DataFrame.
        """
        logger.info("Generating data...")
        if num_rows is None:
            num_rows = self.config.get("data_generation", "num_rows", 1000)
        if columns is None:
            columns = self.config.get("data_generation", "columns")
            if columns is None:
                raise ValueError("Columns configuration is missing.")

        generator = DataGenerator(num_rows=num_rows)
        for column in columns:
            generator.add_column(column['name'], column['data_type'], column.get('options', {}))

        self.dataframe = generator.generate()
        self.dataframe.to_csv(save_path, index=False)
        logger.info(f"Data generated and saved to '{save_path}'.")
        return self.dataframe

    def save_to_sqlite(self, db_path: str = 'data.db', table_name: str = 'generated_data'):
        handler = SQLiteHandler(db_path)
        handler.save_dataframe_to_db(self.dataframe, table_name)
        logger.info(f"Data saved to SQLite database '{db_path}'.")

    def query_data(self, query: str):
        handler = SQLiteHandler(self.db_path)
        handler.create_connection()
        results = handler.execute_query(query)
        logger.info(f"Query executed successfully.")
        handler.close_connection()
        return results


