from utils.logger import logger


class QueryBuilder:
    """
    This module defines the QueryBuilder class, which can be used to build SQL queries.

    Usage:
        query_builder = QueryBuilder()
        query_builder.select("name, age").from_table("users").where("age > 18").build()

    Attributes:
        query (str): The string representation of the SQL query being built.

    Methods:
        __init__(): Initializes the QueryBuilder object.
        select(columns="*"): Adds a SELECT statement to the query.
        from_table(table_name): Adds a FROM statement to the query.
        where(condition): Adds a WHERE statement to the query.
        group_by(columns): Adds a GROUP BY statement to the query.
        having(condition): Adds a HAVING statement to the query.
        order_by(columns, order="ASC"): Adds an ORDER BY statement to the query.
        build(): Finalizes the query and returns the string representation of the query.

    Example:
        query_builder = QueryBuilder()
        query_builder.select("name, age").from_table("users").where("age > 18").build()

    This will generate the following SQL query: "SELECT name, age FROM users WHERE age > 18".
    """
    def __init__(self):
        self.query = ""

    def select(self, columns="*"):
        self.query = f"SELECT {columns} "
        return self

    def from_table(self, table_name):
        self.query += f"FROM {table_name} "
        return self

    def where(self, condition):
        self.query += f"WHERE {condition} "
        return self

    def group_by(self, columns):
        self.query += f"GROUP BY {columns} "
        return self

    def having(self, condition):
        self.query += f"HAVING {condition} "
        return self

    def order_by(self, columns, order="ASC"):
        self.query += f"ORDER BY {columns} {order} "
        return self

    def build(self):
        final_query = self.query.strip()
        self.query = ""
        logger.info(f"Query built: {final_query}")
        return final_query

