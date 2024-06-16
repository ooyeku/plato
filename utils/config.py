import json
from pathlib import Path


class Config:
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.config_data = self.load_config()

    def load_config(self):
        if Path(self.config_file).is_file():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            default_config = self.default_config()
            self.save_config(default_config)
            return default_config

    def save_config(self, config_data):
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=4)

    def default_config(self):
        return {
            "data_ingestion": {
                "csv_loader": {
                    "db_name": "plato-broken.db"
                },
                "crosstab_loader": {
                    "db_name": "plato-broken.db"
                }
            },
            "data_transformation": {
                "cleaner": {
                    "missing_value_strategy": "mean",
                    "duplicate_removal": True
                },
                "transformer": {
                    "label_encoding_columns": [],
                    "one_hot_encoding_columns": [],
                    "scaling_method": "minmax",
                    "scaling_columns": []
                }
            },
            "data_analysis": {
                "qualitative": {
                    "sentiment_analysis_column": "review",
                    "wordcloud_column": "review",
                    "keyword_extraction_column": "review",
                    "keyword_extraction_method": "tfidf",
                    "keyword_extraction_top_n": 5
                },
                "quantitative": {
                    "descriptive_statistics_columns": [],
                    "correlation_matrix_columns": [],
                    "linear_regression_target": "score",
                    "linear_regression_features": ["age", "income"],
                    "hypothesis_testing_columns": ["score", "age"],
                    "hypothesis_testing_method": "t-test",
                    "histogram_column": "income",
                    "histogram_bins": 10,
                    "scatter_plot_columns": ["age", "score"]
                }
            },
            "logging": {
                "level": "INFO"
            },
            "database": {
                "name": "plato-broken.db",
                "path": "./database/"
            }
        }

    def get(self, section, key, default=None):
        return self.config_data.get(section, {}).get(key, default)

    def set(self, section, key, value):
        if section not in self.config_data:
            self.config_data[section] = {}
        self.config_data[section][key] = value
        self.save_config(self.config_data)


# Example usage
if __name__ == "__main__":
    config = Config()
    print(config.get("data_ingestion", "csv_loader"))
    config.set("logging", "level", "DEBUG")
    print(config.get("logging", "level"))
