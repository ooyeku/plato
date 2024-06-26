import pandas as pd
from data_storage.sqlite_handler import SQLiteHandler
from utils.logger import log_info, log_warning, log_error
from concurrent.futures import ThreadPoolExecutor
import os


class CSVLoader:
    """
    CSVLoader Class - Main Object Definition

    The CSVLoader class is a high level utility class that provides design patterns for loading CSV files in a simplified way,
    streamlining the process of reading CSV files into dataframes and storing them into an SQLite database.

    It provides the following methods:

    - __init__(db_name='plato.db', chunk_size=50000): The constructor method for the CSVLoader class. It assigns specified
                                                      database name, chunk size and thread pool executor.

    - load_csv(file_path, table_name=None, save_to_db=False, **kwargs): This method is designed to load a CSV file into a
                                                                        dataframe. It optionally saves the dataframe into
                                                                        a SQLite database.

    - load_multiple_csvs(file_paths, table_names=None, save_to_db=False, **kwargs): This method utilizes multithreading to
                                                                                    load multiple CSV files concurrently.
                                                                                    Optionally, it saves each dataframe into
                                                                                    a different table in the SQLite database.
    """

    def __init__(self, db_name='plato.db', chunk_size=50000):
        self.db_handler = SQLiteHandler(db_name)
        self.chunk_size = chunk_size
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())

    def load_csv(self, file_path, table_name=None, save_to_db=False, **kwargs):
        try:
            if save_to_db:
                if not table_name:
                    table_name = file_path.split('/')[-1].split('.')[0]
                chunk_container = pd.read_csv(file_path, chunksize=self.chunk_size, **kwargs)
                for chunk in chunk_container:
                    self.db_handler.save_dataframe_to_db(chunk, table_name)
                log_info(f"DataFrame saved to table {table_name}")
            else:
                df = pd.read_csv(file_path, **kwargs)
                log_info(f"CSV file loaded from {file_path}")
                return df
        except Exception as e:
            log_error(f"Failed to load CSV file from {file_path}: {e}")
            raise

    def load_multiple_csvs(self, file_paths, table_names=None, save_to_db=False, **kwargs):
        futures = [self.executor.submit(self.load_csv, file_path, table_names[idx] if table_names else None, save_to_db,
                                        **kwargs) for idx, file_path in enumerate(file_paths)]
        log_info("Loading the following CSV files: " + ', '.join(file_paths))
        return [future.result() for future in futures]
