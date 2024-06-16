import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional, Dict, Any


class Visualizer:
    """
     The Visualizer class provides a set of methods to visualize data using plots and charts.

        Attributes:
            df (pd.DataFrame): The DataFrame containing the data.

        Main Methods:
            - plot_histogram: Plot a histogram of a specific column.
            - plot_scatter: Plot a scatter plot between two columns.
            - plot_box: Plot a box plot of a column by another column.
            - plot_heatmap: Plot a heatmap of the correlation matrix.
            - plot_line: Plot a line plot of a column over another column.
            - plot_pie: Plot a pie chart of a column.
            - plot_bar: Plot a bar plot of a column by another column.
            - plot_violin: Plot a violin plot of a column by another column.
            - plot_pairplot: Plot a pairplot of the DataFrame.
            - plot_distribution: Plot a distribution plot of a column.
            - plot_correlation_matrix: Plot a correlation matrix heatmap.
            - plot_3d_scatter: Plot a 3D scatter plot between three columns.
            - plot_facet_grid: Plot a facet grid of scatter plots.
      """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def plot_histogram(self, column: str, bins: int = 10, color: str = 'blue', title: Optional[str] = None,
                       xlabel: Optional[str] = None, ylabel: Optional[str] = None, show: bool = True):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[column], bins=bins, kde=True, color=color)
        plt.title(title or f'Histogram of {column}')
        plt.xlabel(xlabel or column)
        plt.ylabel(ylabel or 'Frequency')
        if show:
            plt.show()

    def plot_scatter(self, x: str, y: str, color: str = 'blue', title: Optional[str] = None,
                     xlabel: Optional[str] = None, ylabel: Optional[str] = None, show: bool = True):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.df[x], y=self.df[y], color=color)
        plt.title(title or f'Scatter plot between {x} and {y}')
        plt.xlabel(xlabel or x)
        plt.ylabel(ylabel or y)
        if show:
            plt.show()

    def plot_box(self, x: str, y: str, color: str = 'blue', title: Optional[str] = None, xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None, show: bool = True):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.df[x], y=self.df[y], color=color)
        plt.title(title or f'Box plot of {y} by {x}')
        plt.xlabel(xlabel or x)
        plt.ylabel(ylabel or y)
        if show:
            plt.show()

    def plot_heatmap(self, correlation: Optional[pd.DataFrame] = None, annot: bool = True, cmap: str = 'coolwarm',
                     title: Optional[str] = None, show: bool = True):
        # Before calling self.df.corr(), we select only the numeric columns
        numeric_df = self.df.select_dtypes(include=[np.number])
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation or numeric_df.corr(), annot=annot, cmap=cmap)
        plt.title(title or 'Heatmap')
        if show:
            plt.show()

    def plot_line(self, x: str, y: str, color: str = 'blue', title: Optional[str] = None, xlabel: Optional[str] = None,
                  ylabel: Optional[str] = None, show: bool = True):
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=self.df[x], y=self.df[y], color=color)
        plt.title(title or f'Line plot of {y} over {x}')
        plt.xlabel(xlabel or x)
        plt.ylabel(ylabel or y)
        if show:
            plt.show()

    def plot_pie(self, names: str, values: str, title: Optional[str] = None, show: bool = True):
        fig = px.pie(self.df, names=names, values=values, title=title)
        if show:
            fig.show()

    def plot_bar(self, x: str, y: str, color: str = 'blue', title: Optional[str] = None, xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None, show: bool = True):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=self.df[x], y=self.df[y], color=color)
        plt.title(title or f'Bar plot of {y} by {x}')
        plt.xlabel(xlabel or x)
        plt.ylabel(ylabel or y)
        if show:
            plt.show()

    def plot_violin(self, x: str, y: str, color: str = 'blue', title: Optional[str] = None,
                    xlabel: Optional[str] = None, ylabel: Optional[str] = None, show: bool = True):
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=self.df[x], y=self.df[y], color=color)
        plt.title(title or f'Violin plot of {y} by {x}')
        plt.xlabel(xlabel or x)
        plt.ylabel(ylabel or y)
        if show:
            plt.show()

    def plot_pairplot(self, hue: Optional[str] = None, palette: str = 'husl', show: bool = True):
        plt.figure(figsize=(10, 8))
        sns.pairplot(self.df, hue=hue, palette=palette)
        plt.title('Pairplot')
        if show:
            plt.show()

    def plot_distribution(self, column: str, color: str = 'blue', title: Optional[str] = None,
                          xlabel: Optional[str] = None, ylabel: Optional[str] = None, show: bool = True):
        plt.figure(figsize=(10, 6))
        sns.displot(self.df[column], color=color)
        plt.title(title or f'Distribution of {column}')
        plt.xlabel(xlabel or column)
        plt.ylabel(ylabel or 'Density')
        if show:
            plt.show()

    def plot_correlation_matrix(self, annot: bool = True, cmap: str = 'coolwarm', title: Optional[str] = None,
                                show: bool = True):
        plt.figure(figsize=(10, 8))
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=annot, cmap=cmap)
        plt.title(title or 'Correlation Matrix')
        if show:
            plt.show()

    def plot_3d_scatter(self, x: str, y: str, z: str, color: str = 'blue', title: Optional[str] = None,
                        show: bool = True):
        fig = px.scatter_3d(self.df, x=x, y=y, z=z, color=color, title=title)
        if show:
            fig.show()

    def plot_facet_grid(self, row: str, col: str, x: str, y: str, hue: Optional[str] = None, show: bool = True):
        g = sns.FacetGrid(self.df, row=row, col=col, hue=hue)
        g.map(sns.scatterplot, x, y)
        plt.title('Facet Grid')
        if show:
            plt.show()


# Example usage
if __name__ == "__main__":
    data = {
        'score': [5, 1, 5, 2, 4],
        'age': [25, 30, 35, 40, 45],
        'income': [50000, 60000, 70000, 80000, 90000],
        'category': ['A', 'B', 'A', 'B', 'A']
    }
    df = pd.DataFrame(data)
    visualizer = Visualizer(df)
    visualizer.plot_histogram(column='age', bins=5, color='green')
    visualizer.plot_scatter(x='age', y='income', color='red')
    visualizer.plot_box(x='category', y='income', color='blue')
    visualizer.plot_heatmap()
    visualizer.plot_line(x='age', y='income', color='purple')
    visualizer.plot_pie(names='category', values='score')
    visualizer.plot_bar(x='category', y='income', color='orange')
    visualizer.plot_violin(x='category', y='income', color='cyan')
    visualizer.plot_pairplot(hue='category')
    visualizer.plot_distribution(column='income', color='magenta')
    visualizer.plot_correlation_matrix()
    visualizer.plot_3d_scatter(x='age', y='income', z='score', color='category')
    visualizer.plot_facet_grid(row='category', col='age', x='score', y='income')
