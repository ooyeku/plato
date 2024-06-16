import pandas as pd
import numpy as np
from faker import Faker
from joblib import Parallel, delayed


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
