import pandas as pd
import numpy as np
from utils.logger import logger
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# Configure logging
logger.setLevel("INFO")


class Modeler:
    def __init__(self, df):
        self.df = df.copy()

    def train_test_split(self, target, features, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.

        Parameters:
            target (str): The target column.
            features (list): List of feature columns.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Seed used by the random number generator.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X = self.df[features]
        y = self.df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logger.info(f"Data split into train and test sets with test_size={test_size}")
        return X_train, X_test, y_train, y_test

    def linear_regression(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate a linear regression model.

        Returns:
            dict: Model, predictions, MSE, R2 score
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        logger.info("Linear regression model trained and evaluated")
        return {
            'model': model,
            'predictions': predictions,
            'mse': mse,
            'r2': r2
        }

    def logistic_regression(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate a logistic regression model.

        Returns:
            dict: Model, predictions, accuracy, classification report
        """
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        logger.info("Logistic regression model trained and evaluated")
        return {
            'model': model,
            'predictions': predictions,
            'accuracy': accuracy,
            'report': report
        }

    def decision_tree_classifier(self, X_train, X_test, y_train, y_test, params=None):
        """
        Train and evaluate a decision tree classifier.

        Returns:
            dict: Model, predictions, accuracy, classification report
        """
        if params:
            model = GridSearchCV(DecisionTreeClassifier(), params, cv=5)
        else:
            model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        logger.info("Decision tree classifier trained and evaluated")
        return {
            'model': model,
            'predictions': predictions,
            'accuracy': accuracy,
            'report': report
        }

    def decision_tree_regressor(self, X_train, X_test, y_train, y_test, params=None):
        """
        Train and evaluate a decision tree regressor.

        Returns:
            dict: Model, predictions, MSE, R2 score
        """
        if params:
            model = GridSearchCV(DecisionTreeRegressor(), params, cv=5)
        else:
            model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        logger.info("Decision tree regressor trained and evaluated")
        return {
            'model': model,
            'predictions': predictions,
            'mse': mse,
            'r2': r2
        }

    def random_forest_classifier(self, X_train, X_test, y_train, y_test, params=None):
        """
        Train and evaluate a random forest classifier.

        Returns:
            dict: Model, predictions, accuracy, classification report
        """
        if params:
            model = GridSearchCV(RandomForestClassifier(), params, cv=5)
        else:
            model = RandomForestClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        logger.info("Random forest classifier trained and evaluated")
        return {
            'model': model,
            'predictions': predictions,
            'accuracy': accuracy,
            'report': report
        }

    def random_forest_regressor(self, X_train, X_test, y_train, y_test, params=None):
        """
        Train and evaluate a random forest regressor.

        Returns:
            dict: Model, predictions, MSE, R2 score
        """
        if params:
            model = GridSearchCV(RandomForestRegressor(), params, cv=5)
        else:
            model = RandomForestRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        logger.info("Random forest regressor trained and evaluated")
        return {
            'model': model,
            'predictions': predictions,
            'mse': mse,
            'r2': r2
        }
