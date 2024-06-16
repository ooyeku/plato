import pandas as pd
import numpy as np
from utils.logger import logger
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Setting up logging
logger.setLevel("INFO")


class QualitativeAnalysis:
    """
    The QualitativeAnalysis class provides a set of methods to perform qualitative analysis on text data.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the text data.

    Main Methods:
        - sentiment_analysis: Perform sentiment analysis on a text column.
        - generate_wordcloud: Generate a word cloud from a text column.
        - keyword_extraction: Extract keywords from a text column.
        - get_qualitative_data: Get the DataFrame with qualitative analysis results.

    Remarks:
        - The sentiment analysis method uses the VADER sentiment analysis tool.
        - The keyword extraction method supports both TF-IDF and count-based methods.
    """
    def __init__(self, df):
        self.df = df.copy()

    def sentiment_analysis(self, text_column):
        """
        Perform sentiment analysis on a text column.

        Parameters:
            text_column (str): The column containing text data.

        Returns:
            pd.DataFrame: DataFrame with sentiment scores.
        """
        sid = SentimentIntensityAnalyzer()
        self.df['sentiment'] = self.df[text_column].apply(lambda x: sid.polarity_scores(x)['compound'])
        logger.info(f"Sentiment analysis performed on column: {text_column}")
        return self.df

    def generate_wordcloud(self, text_column, max_words=100, background_color="white"):
        """
        Generate a word cloud from a text column.

        Parameters:
            text_column (str): The column containing text data.
            max_words (int): Maximum number of words in the word cloud.
            background_color (str): Background color for the word cloud.

        Returns:
            None
        """
        text = ' '.join(self.df[text_column].dropna())
        wordcloud = WordCloud(max_words=max_words, background_color=background_color).generate(text)
        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        logger.info(f"Word cloud generated for column: {text_column}")

    def keyword_extraction(self, text_column, method='tfidf', top_n=10):
        """
        Extract keywords from a text column.

        Parameters:
            text_column (str): The column containing text data.
            method (str): Method to use for keyword extraction ('tfidf' or 'count').
            top_n (int): Number of top keywords to extract.

        Returns:
            pd.DataFrame: DataFrame with keywords and their scores.
        """
        if method == 'tfidf':
            vectorizer = TfidfVectorizer(stop_words='english')
        elif method == 'count':
            vectorizer = CountVectorizer(stop_words='english')
        else:
            raise ValueError(f"Unknown method: {method}")

        X = vectorizer.fit_transform(self.df[text_column].dropna())
        scores = np.sum(X.toarray(), axis=0)
        keywords = vectorizer.get_feature_names_out()
        keyword_scores = pd.DataFrame({'keyword': keywords, 'score': scores}).sort_values(by='score',
                                                                                          ascending=False).head(top_n)
        logger.info(f"Keyword extraction performed on column: {text_column} using {method} method")
        return keyword_scores

    def get_qualitative_data(self):
        """
        Get the DataFrame with qualitative analysis results.

        Returns:
            pd.DataFrame: The DataFrame with qualitative analysis results.
        """
        return self.df
