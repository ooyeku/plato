import pandas as pd
import numpy as np
from faker import Faker
from joblib import Parallel, delayed


"""
DataGenerator class in Python is used to generate fake data based on specified requirements. It's often used for testing, populating databases for
development environments, etc.

Attributes:
    num_rows (int): Defines the number of rows for the fake data to be created.
    fake (Faker): An instance of the Faker module, responsible for generating fake data.
    data (pd.DataFrame): The generated fake data stored as a pandas DataFrame.

Functions:
    __init__(): Initializes the DataGenerator instance with the given number of rows.
    add_column(): Adds a new column to the DataFrame with fake data values based on the given data type and options.
    generate(): Returns the generated DataFrame with the fake data.

Note:
    Specifications for each fake data type (integer, email, date, etc.) and the corresponding data generation mechanisms are defined in
    the add_column() function. Modifications or additions to data types should be done in this function.
"""

class DataGenerator:
    def __init__(self, num_rows):
        self.num_rows = num_rows
        self.fake = Faker()
        self.data = pd.DataFrame(index=range(num_rows))


    def add_column(self, column_name, data_type, options=None):
        if column_name == 'age':
            self.data[column_name] = [self.fake.random_int(min=18, max=100) for _ in range(self.num_rows)]
        elif data_type == 'email':
            self.data[column_name] = [self.fake.email() for _ in range(self.num_rows)]
        elif data_type == 'date':
            start_date = pd.to_datetime(options['min'])
            end_date = pd.to_datetime(options['max'])
            self.data[column_name] = start_date + (end_date - start_date) * np.random.rand(self.num_rows)
        elif data_type == 'category':
            self.data[column_name] = np.random.choice(options['categories'], self.num_rows)
        elif data_type == 'bool':
            self.data[column_name] = np.random.choice([True, False], self.num_rows)
        elif data_type == 'float':
            self.data[column_name] = np.random.uniform(options['min'], options['max'], self.num_rows)
        elif data_type == 'text':
            self.data[column_name] = [self.fake.text(max_nb_chars=options['max_chars']) for _ in range(self.num_rows)]

    def generate(self):
        return self.data
