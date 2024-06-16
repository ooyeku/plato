import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
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
        le = LabelEncoder()
        for column in columns:
            self.df[column] = le.fit_transform(self.df[column])
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
        # check if 'columns' is a string (single column), if so convert it to a list
        if isinstance(columns, str):
            columns = [columns]

        self.df = pd.get_dummies(self.df, columns=columns)
        logger.info(f"One-hot encoding applied to columns: {columns}")
        return self

    def scale_data(self, columns, method='minmax'):
        for col in columns:
            if method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown method: {method}")

            self.df[col] = scaler.fit_transform(self.df[col].values.reshape(-1, 1))
            logger.info(f"Data scaled using {method} method for column: {col}")
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

    # def polynomial_features(self, columns, degree=2, interaction_only=False, include_bias=False):
    #     """
    #     Generate polynomial features.
    #
    #     Parameters:
    #         columns (list or str): Columns to generate polynomial features for.
    #         degree (int): The degree of the polynomial features.
    #         interaction_only (bool): Whether to include only interaction features.
    #         include_bias (bool): Whether to include a bias column.
    #
    #     Returns:
    #         DataTransformer: self (to allow method chaining).
    #     """
    #     from sklearn.preprocessing import PolynomialFeatures
    #
    #     poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    #
    #     # Check if columns is a string (single column), convert to dataframe and ensure it's 2-D
    #     if isinstance(columns, str):
    #         poly_features = poly.fit_transform(self.df[[columns]])
    #     else:
    #         poly_features = poly.fit_transform(self.df[columns])
    #
    # poly_columns = poly.get_feature_names_out(columns) self.df = pd.concat([self.df.drop(columns, axis=1),
    # pd.DataFrame(poly_features, columns=poly_columns)], axis=1) logger.info(f"Polynomial features generated for
    # columns: {columns} with degree {degree}") return self

    def get_transformed_data(self):
        """
        Get the transformed DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        return self.df
