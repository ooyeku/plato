import pandas as pd
import numpy as np
from utils.logger import logger

# Setting up logging
logger.setLevel("INFO")

"""
DataCleaner Class
-----------------

This class is designed for cleaning and pre-processing DataFrame objects. The following cleaning operations are implemented -
duplicate removal, filling missing values, dropping rows with missing values, replacing specific values, removing outliers,
converting datatype of columns, normalizing and standardizing data.

Methods
-------
- __init__: Constructor takes in the input DataFrame and creates a copy for cleaning.
- remove_duplicates(self, subset=None, keep='first'): This method removes duplicate rows based on some subset of columns.
    By default, it keeps the first occurrence of the duplicate.
- fill_missing_values(self, strategy='mean', columns=None): This method fills missing values with mean, median, mode or any
    specified value (provided as strategy). By default, it is applied to all columns.
- drop_missing_values(self, columns=None, how='any'): This method drops rows with missing values. It allows selection of
    specific columns and dropping strategy (any or all).
- replace_values(self, to_replace, value, columns=None): This method replaces specific values in selected columns.
- remove_outliers(self, columns=None, method='IQR', factor=1.5): This method removes outliers from the DataFrame using either
    the IQR method or the Z-score method.
- convert_data_types(self, columns, target_type, **kwargs): This method converts data types of the specified columns to the
    target data type. Additionally, dtype-specific keyword arguments can be provided.
- normalize_data(self, columns=None): This method scales the specified columns of the DataFrame so that they have a minimum of
    0 and a maximum of 1.
- standardize_data(self, columns=None): This method standardizes the specified columns of the DataFrame so they have a mean of
    0 and a standard deviation of 1.
- get_cleaned_data(self): This method returns the cleaned DataFrame.

Log messages are generated after each cleaning operation to track the changes to the data.
"""


class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()

    def remove_duplicates(self, subset=None, keep='first'):
        """
        Remove duplicate rows from the DataFrame.

        Parameters:
            subset (list or str): Columns to consider for identifying duplicates.
            keep (str): Which duplicates to keep ('first', 'last', or False).

        Returns:
            DataCleaner: self (to allow method chaining).
        """
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        logger.info("Duplicates removed")
        return self

    def fill_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.columns

        if strategy in ['mean', 'median']:
            self.df[columns] = self.df[columns].apply(
                lambda x: x.fillna(x.mean()) if pd.api.types.is_numeric_dtype(x) else x)
        elif strategy == 'mode':
            self.df[columns] = self.df[columns].apply(lambda x: x.fillna(x.mode()[0]))
        else:
            self.df[columns].fillna(strategy, inplace=True)

        return self

    def drop_missing_values(self, columns=None, how='any'):
        """
        Drop rows with missing values.

        Parameters:
            columns (list or str): Specific columns to check for missing values. If None, checks all columns.
            how (str): Whether to drop rows with any or all missing values ('any' or 'all').

        Returns:
            DataCleaner: self (to allow method chaining).
        """
        self.df.dropna(subset=columns, how=how, inplace=True)
        logger.info(f"Rows with missing values dropped ({how})")
        return self

    def replace_values(self, to_replace, value, columns=None):
        if columns is None:
            columns = self.df.columns
        self.df[columns] = self.df[columns].replace(to_replace, value)
        return self

    def remove_outliers(self, columns=None, method='IQR', factor=1.5):
        """
        Remove outliers from the DataFrame.

        Parameters:
            columns (list or str): Specific columns to check for outliers. If None, applies to all numeric columns.
            method (str): Method to use for outlier detection ('IQR' or 'Z-score').
            factor (float): Factor for the method used. For 'IQR', it multiplies the IQR. For 'Z-score', it is the threshold.

        Returns:
            DataCleaner: self (to allow method chaining).
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        if method == 'IQR':
            Q1 = self.df[columns].quantile(0.25)
            Q3 = self.df[columns].quantile(0.75)
            IQR = Q3 - Q1

            if isinstance(columns, list):
                self.df = self.df[
                    ~((self.df[columns] < (Q1 - factor * IQR)) | (self.df[columns] > (Q3 + factor * IQR))).any(axis=1)]
            else:  # Modify this section to treat Series.
                self.df = self.df[
                    ~((self.df[columns] < (Q1 - factor * IQR)) | (self.df[columns] > (Q3 + factor * IQR)))]

        elif method == 'Z-score':
            from scipy import stats
            z_scores = np.abs(stats.zscore(self.df[columns]))
            self.df = self.df[(z_scores < factor).all(axis=1)]
        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"Outliers removed using {method} method")
        return self

    def convert_data_types(self, columns, target_type, **kwargs):
        """
        Convert the data type of specified columns.

        Args:
            columns (list): The columns to convert.
            target_type (str): The target data type to convert to.
            **kwargs: Additional keyword arguments for the conversion method.

        Returns:
            DataCleaner: self (to allow method chaining).
        """
        for column in columns:
            if target_type == 'datetime':
                self.df[column] = pd.to_datetime(self.df[column], **kwargs)
            else:
                self.df[column] = self.df[column].astype(target_type)
        return self

    def normalize_data(self, columns=None):
        """
        Normalize data in the DataFrame.

        Parameters:
            columns (list or str): Specific columns to normalize. If None, normalizes all numeric columns.

        Returns:
            DataCleaner: self (to allow method chaining).
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        self.df[columns] = (self.df[columns] - self.df[columns].min()) / (
                self.df[columns].max() - self.df[columns].min())
        logger.info(f"Data normalized for columns: {columns}")
        return self

    def standardize_data(self, columns=None):
        """
        Standardize data in the DataFrame.

        Parameters:
            columns (list or str): Specific columns to standardize. If None, standardizes all numeric columns.

        Returns:
            DataCleaner: self (to allow method chaining).
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        self.df[columns] = (self.df[columns] - self.df[columns].mean()) / self.df[columns].std()
        logger.info(f"Data standardized for columns: {columns}")
        return self

    def get_cleaned_data(self):
        """
        Get the cleaned DataFrame.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        return self.df
