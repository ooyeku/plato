import unittest
from data_storage import query_builder


class TestQueryBuilder(unittest.TestCase):
    def setUp(self):
        self.query_builder = query_builder.QueryBuilder()

    def test_select(self):
        self.query_builder.select("name, age")
        self.assertEqual(self.query_builder.query, "SELECT name, age ")

    def test_from_table(self):
        self.query_builder.from_table("users")
        self.assertEqual(self.query_builder.query, "FROM users ")

    def test_where(self):
        self.query_builder.where("age > 18")
        self.assertEqual(self.query_builder.query, "WHERE age > 18 ")

    def test_group_by(self):
        self.query_builder.group_by("age")
        self.assertEqual(self.query_builder.query, "GROUP BY age ")

    def test_having(self):
        self.query_builder.having("COUNT(name) > 1")
        self.assertEqual(self.query_builder.query, "HAVING COUNT(name) > 1 ")

    def test_order_by(self):
        self.query_builder.order_by("age", "DESC")
        self.assertEqual(self.query_builder.query, "ORDER BY age DESC ")

    def test_build(self):
        self.query_builder.select("name, age").from_table("users").where("age > 18").build()
        self.assertEqual(self.query_builder.query, "")


if __name__ == '__main__':
    unittest.main()