import pandas as pd
import numpy as np
import logging
from category_encoders import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTransformer:
    def __init__(self, df):
        self.df = df.copy()

    def encode_labels(self, columns):
        """
        Encode labels in specific columns using label encoding.

        Parameters:
            columns (list or str): Columns to label encode.

        Returns:
            DataTransformer: self (to allow method chaining).
        """
        encoder = OrdinalEncoder(cols=columns)
        self.df = encoder.fit_transform(self.df)
        logger.info(f"Labels encoded for columns: {columns}")
        return self

    def one_hot_encode(self, columns):
        """
        Apply one-hot encoding to specific columns.

        Parameters:
            columns (list or str): Columns to one-hot encode.

        Returns:
            DataTransformer: self (to allow method chaining).
        """
        encoder = OneHotEncoder(cols=columns)
        self.df = encoder.fit_transform(self.df)
        logger.info(f"One-hot encoding applied to columns: {columns}")
        return self

    def scale_data(self, columns, method='minmax'):
        scaler = MinMaxScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        logger.info(f"Data scaled using {method} method for columns: {columns}")
        return self

    def log_transform(self, columns):
        """
        Apply log transformation to specific columns.

        Parameters:
            columns (list or str): Columns to log transform.

        Returns:
            DataTransformer: self (to allow method chaining).
        """
        self.df[columns] = np.log1p(self.df[columns])
        logger.info(f"Log transformation applied to columns: {columns}")
        return self

    def apply_custom_transform(self, columns, func):
        """
        Apply a custom transformation function to specific columns.

        Parameters:
            columns (list or str): Columns to apply the transformation to.
            func (function): Custom function to apply.

        Returns:
            DataTransformer: self (to allow method chaining).
        """
        self.df[columns] = self.df[columns].apply(func)
        logger.info(f"Custom transformation applied to columns: {columns}")
        return self

    def bin_data(self, columns, bins, labels=None):
        """
        Bin data into discrete intervals.

        Parameters:
            columns (list or str): Columns to bin.
            bins (int or list): The criteria to bin the data.
            labels (list or str): Labels for the bins.

        Returns:
            DataTransformer: self (to allow method chaining).
        """
        self.df[columns] = pd.cut(self.df[columns], bins=bins, labels=labels)
        logger.info(f"Data binned for columns: {columns}")
        return self

    def get_transformed_data(self):
        """
        Get the transformed DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        return self.df