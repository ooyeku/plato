import pandas as pd
import numpy as np
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

        for column in columns:
            if pd.api.types.is_numeric_dtype(self.df[column]):  # check if column is numeric
                if strategy == 'mean':
                    self.df[column].fillna(self.df[column].mean(), inplace=True)
                elif strategy == 'median':
                    self.df[column].fillna(self.df[column].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[column].fillna(self.df[column].mode()[0], inplace=True)
                else:
                    self.df[column].fillna(strategy, inplace=True)
            else:  # if column is non-numeric (e.g., object or categorical type)
                if strategy == 'mode':
                    self.df[column].fillna(self.df[column].mode()[0], inplace=True)
                else:
                    self.df[column].fillna(strategy, inplace=True)

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
        """
        Replace values in the DataFrame.

        Parameters:
            to_replace (any): Values to replace.
            value (any): Value to replace with.
            columns (list or str): Specific columns to apply the replacement. If None, applies to all columns.

        Returns:
            DataCleaner: self (to allow method chaining).
        """
        self.df.replace(to_replace, value, inplace=True)
        logger.info(f"Values replaced: {to_replace} -> {value}")
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
                self.df = self.df[~((self.df[columns] < (Q1 - factor * IQR)) | (self.df[columns] > (Q3 + factor * IQR)))]

        elif method == 'Z-score':
            from scipy import stats
            z_scores = np.abs(stats.zscore(self.df[columns]))
            self.df = self.df[(z_scores < factor).all(axis=1)]
        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"Outliers removed using {method} method")
        return self

    def convert_data_types(self, columns, dtype):
        """
        Convert data types of specific columns.

        Parameters:
            columns (list or str): Columns to convert.
            dtype (type): Data type to convert to.

        Returns:
            DataCleaner: self (to allow method chaining).
        """
        if dtype == 'datetime':
            for column in columns:
                self.df[column] = pd.to_datetime(self.df[column])
        else:
            self.df[columns] = self.df[columns].astype(dtype)
        logger.info(f"Data types converted for columns: {columns} to {dtype}")
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

        self.df[columns] = (self.df[columns] - self.df[columns].min()) / (self.df[columns].max() - self.df[columns].min())
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