import pandas as pd
import numpy as np
from faker import Faker
import random
from typing import List, Dict, Any, Optional

import data_ingestion

fake = Faker()


class DataGenerator:
    def __init__(self, num_rows: int):
        self.num_rows = num_rows
        self.columns = {}

    def add_column(self, name: str, data_type: str, options: Optional[Dict[str, Any]] = None):
        """
        Add a column definition to the generator.

        Parameters:
            name (str): The name of the column.
            data_type (str): The type of data to generate ('int', 'float', 'str', 'date', 'category', 'bool', 'email', 'phone', 'address', 'name', 'company', etc.).
            options (dict, optional): Additional options for data generation.
        """
        self.columns[name] = (data_type, options or {})

    def generate(self) -> pd.DataFrame:
        """
        Generate the data based on the column definitions.

        Returns:
            pd.DataFrame: The generated DataFrame.
        """
        data = {}
        for column, (data_type, options) in self.columns.items():
            data[column] = self._generate_column_data(data_type, options)
        return pd.DataFrame(data)

    def _generate_column_data(self, data_type: str, options: Dict[str, Any]) -> List:
        """
        Generate data for a single column.

        Parameters:
            data_type (str): The type of data to generate.
            options (dict): Additional options for data generation.

        Returns:
            list: The generated data for the column.
        """
        if data_type == 'int':
            return [random.randint(options.get('min', 0), options.get('max', 100)) for _ in range(self.num_rows)]
        elif data_type == 'float':
            return [random.uniform(options.get('min', 0), options.get('max', 100)) for _ in range(self.num_rows)]
        elif data_type == 'str':
            return [fake.word() for _ in range(self.num_rows)]
        elif data_type == 'date':
            start_date = options.get('start_date', '-30y')
            end_date = options.get('end_date', 'now')
            return [fake.date_between(start_date=start_date, end_date=end_date) for _ in range(self.num_rows)]
        elif data_type == 'category':
            return [random.choice(options.get('categories', ['A', 'B', 'C'])) for _ in range(self.num_rows)]
        elif data_type == 'bool':
            return [random.choice([True, False]) for _ in range(self.num_rows)]
        elif data_type == 'email':
            return [fake.email() for _ in range(self.num_rows)]
        elif data_type == 'phone':
            return [fake.phone_number() for _ in range(self.num_rows)]
        elif data_type == 'address':
            return [fake.address() for _ in range(self.num_rows)]
        elif data_type == 'name':
            return [fake.name() for _ in range(self.num_rows)]
        elif data_type == 'company':
            return [fake.company() for _ in range(self.num_rows)]
        elif data_type == 'text':
            return [fake.text(max_nb_chars=options.get('max_nb_chars', 200)) for _ in range(self.num_rows)]
        else:
            raise ValueError(f"Unsupported data type: {data_type}")


# Example usage
if __name__ == "__main__":
    generator = DataGenerator(num_rows=100000)
    generator.add_column('id', 'int', {'min': 1, 'max': 1000})
    generator.add_column('name', 'name')
    generator.add_column('age', 'int', {'min': 18, 'max': 90})
    generator.add_column('email', 'email')
    generator.add_column('join_date', 'date', {'start_date': '-5y', 'end_date': 'now'})
    generator.add_column('category', 'category', {'categories': ['A', 'B', 'C', 'D']})
    generator.add_column('is_active', 'bool')
    generator.add_column('salary', 'float', {'min': 30000, 'max': 120000})

    df = generator.generate()
    print(df.head())
    print(df.info())
