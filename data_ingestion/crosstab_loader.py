import pandas as pd
from utils.logger import logger
from data_storage.sqlite_handler import SQLiteHandler

# Setting up logging
logger.setLevel("INFO")


class CrosstabLoader:
    """
    The CrosstabLoader class provides a set of methods to load crosstabs from Excel files into pandas DataFrames and optionally save them to a SQLite database.

    Attributes:
        db_handler (SQLiteHandler instance): An instance of SQLiteHandler to interact with the SQLite database.

    Main Methods:
        - __init__: Initialize CrosstabLoader with an optional SQLite database name. Defaults to 'plato.db'.
        - load_crosstab: Load a crosstab from an Excel file into a DataFrame and optionally save it to the database.
        - load_multiple_crosstabs: Load crosstabs from multiple Excel files into DataFrames and optionally save them to the database.
        - load_crosstab_to_db: Load a crosstab directly to the database from an Excel file without returning a DataFrame.

    Remarks:
        - The load_crosstab method returns a DataFrame if one sheet is loaded from the file or a dictionary of DataFrames if multiple sheets are loaded.
        - The load_multiple_crosstabs returns a list of DataFrames or dictionaries of DataFrames.
        - The **kwargs in load_crosstab, load_multiple_crosstabs, and load_crosstab_to_db can be used to pass any additional parameters to pd.read_excel.
    """

    def __init__(self, db_name='plato.db'):
        self.db_handler = SQLiteHandler(db_name)

    def load_crosstab(self, file_path, sheet_name=None, table_name=None, save_to_db=False, **kwargs):
        """
        Load a crosstab from an Excel file into a pandas DataFrame and optionally save it to the database.

        Parameters:
            file_path (str): The path to the Excel file.
            sheet_name (str or None): The sheet name to load. If None, loads all sheets.
            table_name (str): The name of the table to save to. If None, uses the sheet name or file name.
            save_to_db (bool): Whether to save the DataFrame to the database.
            kwargs: Additional keyword arguments to pass to pd.read_excel.

        Returns:
            pd.DataFrame or dict: The loaded DataFrame or a dictionary of DataFrames if multiple sheets are loaded.
        """
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            logger.info(f"Crosstab loaded from {file_path} (sheet: {sheet_name})")

            if save_to_db:
                if not table_name:
                    table_name = sheet_name if sheet_name else file_path.split('/')[-1].split('.')[0]
                self.db_handler.save_dataframe_to_db(df, table_name)
                logger.info(f"DataFrame saved to table {table_name}")

            return df
        else:
            sheets = pd.read_excel(file_path, sheet_name=None, **kwargs)
            logger.info(f"All sheets loaded from {file_path}")

            if save_to_db:
                for name, df in sheets.items():
                    table_name = name if not table_name else table_name
                    self.db_handler.save_dataframe_to_db(df, table_name)
                    logger.info(f"DataFrame saved to table {table_name}")

            return sheets

    def load_multiple_crosstabs(self, file_paths, sheet_names=None, table_names=None, save_to_db=False, **kwargs):
        """
        Load multiple crosstabs from Excel files into pandas DataFrames and optionally save them to the database.

        Parameters:
            file_paths (list): List of paths to the Excel files.
            sheet_names (list): List of sheet names to load. If None, loads all sheets.
            table_names (list): List of table names to save to. If None, uses sheet names or file names.
            save_to_db (bool): Whether to save the DataFrames to the database.
            kwargs: Additional keyword arguments to pass to pd.read_excel.

        Returns:
            list: List of loaded DataFrames or dictionaries of DataFrames if multiple sheets are loaded.
        """
        dataframes = []
        for idx, file_path in enumerate(file_paths):
            sheet_name = sheet_names[idx] if sheet_names else None
            table_name = table_names[idx] if table_names else None
            df = self.load_crosstab(file_path, sheet_name, table_name, save_to_db, **kwargs)
            dataframes.append(df)
        return dataframes

    def load_crosstab_to_db(self, file_path, sheet_name=None, table_name=None, **kwargs):
        """
        Load a crosstab from an Excel file directly to the database without returning a DataFrame.

        Parameters:
            file_path (str): The path to the Excel file.
            sheet_name (str or None): The sheet name to load. If None, loads all sheets.
            table_name (str): The name of the table to save to. If None, uses the sheet name or file name.
            kwargs: Additional keyword arguments to pass to pd.read_excel.
        """
        self.load_crosstab(file_path, sheet_name, table_name, save_to_db=True, **kwargs)
