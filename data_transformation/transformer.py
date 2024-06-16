import pandas as pd
import numpy as np
from category_encoders import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from utils.logger import logger

# Setting up logging
logger.setLevel("INFO")


class DataTransformer:
    """
    The DataTransformer class acts as a container for various data transformation operations that could be performed
    on a pandas DataFrame. This is designed to be an easy-to-use, chainable interface for data preprocessing tasks that
    is built on the Pandas DataFrame structure. The DataTransformer class structure makes it possible to complete
    several transformation tasks in a single line of code, treating the transformations as methods that modify the
    instance's DataFrame.

    Instance Attributes:
        df (pd.DataFrame): The DataFrame to process. It is a copy of the original DataFrame to prevent any unwanted
            alterations to the original data.

    Class Methods:
        __init__(self, df): Initializes the instance and creates a copy of the input DataFrame.

        encode_labels(self, columns): Label encodes the data of the specified columns in the DataFrame.

        one_hot_encode(self, columns): Performs one-hot encoding on the data of the specified columns in the DataFrame.

        scale_data(self, columns, method='minmax'): Scales data within the specified 'columns' using the provided
            'method'. By default, this is set to the 'minmax' method.

        log_transform(self, columns): Applies a log transformation to the data in the specified columns of the DataFrame.

        apply_custom_transform(self, columns, func): Applies a custom transformation function 'func' to the data of
            specified 'columns' in the DataFrame.

        bin_data(self, columns, bins, labels=None): Bins data into discrete intervals in the specified 'columns' of
            the DataFrame using the criteria defined in 'bins'. Labels for the bins can be optionally provided.

        get_transformed_data(self): Returns the transformed DataFrame for use in further processing or machine learning
            tasks.

    Shared method parameters:
        Method parameters of interest include 'columns', often a list of column names. These values dictate which columns
        the transformation should be applied to. In many methods, these transformations can be chained, enabling
        sequential transformation using a single line of code.

    Logging:
        In every method apart from the constructor, logging is performed to indicate the success of the transformation
        and record the columns that have been transformed. This is useful in both error tracking and retracing the
        transformation steps.
    """

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
