import pandas as pd
from data_storage.sqlite_handler import SQLiteHandler
from utils.logger import log_info, log_warning, log_error


class CSVLoader:
    def __init__(self, db_name='plato-broken.db'):
        self.db_handler = SQLiteHandler(db_name)

    def load_csv(self, file_path, table_name=None, save_to_db=False, **kwargs):
        try:
            df = pd.read_csv(file_path, **kwargs)
            log_info(f"CSV file loaded from {file_path}")

            if save_to_db:
                if not table_name:
                    table_name = file_path.split('/')[-1].split('.')[0]
                self.db_handler.save_dataframe_to_db(df, table_name)
                log_info(f"DataFrame saved to table {table_name}")

            return df
        except Exception as e:
            log_error(f"Failed to load CSV file from {file_path}: {e}")
            raise

    def load_multiple_csvs(self, file_paths, table_names=None, save_to_db=False, **kwargs):
        dataframes = []
        for idx, file_path in enumerate(file_paths):
            table_name = table_names[idx] if table_names else None
            df = self.load_csv(file_path, table_name, save_to_db, **kwargs)
            dataframes.append(df)
        return dataframes

    def load_csv_to_db(self, file_path, table_name=None, **kwargs):
        self.load_csv(file_path, table_name, save_to_db=True, **kwargs)

