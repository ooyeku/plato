import pandas as pd
import numpy as np
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantitativeAnalysis:
    def __init__(self, df):
        self.df = df.copy()

    def descriptive_statistics(self):
        """
        Calculate descriptive statistics for the DataFrame.

        Returns:
            pd.DataFrame: DataFrame with descriptive statistics.
        """
        desc_stats = self.df.describe()
        logger.info("Descriptive statistics calculated")
        return desc_stats

    def correlation_matrix(self):
        """
        Calculate the correlation matrix for the DataFrame.

        Returns:
            pd.DataFrame: DataFrame with correlation matrix.
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        logger.info("Correlation matrix calculated")
        return corr_matrix

    def plot_correlation_matrix(self):
        """
        Plot the correlation matrix.

        Returns:
            None
        """
        corr_matrix = self.correlation_matrix()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()
        logger.info("Correlation matrix plotted")

    def linear_regression(self, target, features):
        """
        Perform linear regression.

        Parameters:
            target (str): The target column.
            features (list): List of feature columns.

        Returns:
            dict: Dictionary with model, predictions, and performance metrics.
        """
        X = self.df[features]
        y = self.df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        logger.info("Linear regression performed")
        return {
            'model': model,
            'predictions': predictions,
            'mse': mse,
            'r2': r2
        }

    def hypothesis_testing(self, column1, column2, test='t-test'):
        """
        Perform hypothesis testing between two columns.

        Parameters:
            column1 (str): The first column.
            column2 (str): The second column.
            test (str): The type of test to perform ('t-test' or 'anova').

        Returns:
            dict: Dictionary with test results.
        """
        if test == 't-test':
            t_stat, p_value = stats.ttest_ind(self.df[column1], self.df[column2], nan_policy='omit')
        elif test == 'anova':
            f_stat, p_value = stats.f_oneway(self.df[column1], self.df[column2])
        else:
            raise ValueError(f"Unknown test: {test}")

        logger.info(f"Hypothesis testing performed: {test} between {column1} and {column2}")
        return {
            'statistic': t_stat if test == 't-test' else f_stat,
            'p_value': p_value
        }

    def plot_histogram(self, column, bins=10):
        """
        Plot a histogram of a specific column.

        Parameters:
            column (str): The column to plot.
            bins (int): Number of bins.

        Returns:
            None
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[column], bins=bins, kde=True)
        plt.title(f'Histogram of {column}')
        plt.show()
        logger.info(f"Histogram plotted for column: {column}")

    def plot_scatter(self, column1, column2):
        """
        Plot a scatter plot between two columns.

        Parameters:
            column1 (str): The first column.
            column2 (str): The second column.

        Returns:
            None
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.df[column1], y=self.df[column2])
        plt.title(f'Scatter plot between {column1} and {column2}')
        plt.show()
        logger.info(f"Scatter plot plotted between {column1} and {column2}")

    def get_quantitative_data(self):
        """
        Get the DataFrame with quantitative analysis results.

        Returns:
            pd.DataFrame: The DataFrame with quantitative analysis results.
        """
        return self.df