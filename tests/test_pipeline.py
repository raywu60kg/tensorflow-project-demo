from src.pipeline import PostgreSQL2Tfrecord
import os

package_dir = os.path.dirname(os.path.abspath(__file__))

sql2tfrecord = PostgreSQL2Tfrecord()
test_data = {
    "sepal_length": [5.1, 7, 6.3],
    "sepal_width": [3.5, 3.2, 3.3],
    "petal_length": [1.4, 4.7, 6],
    "petal_width": [0.2, 1.4, 2.5],
    "variety": ["Setosa", "Versicolor", "Virginica"]
}
test_format_data = {
    "sepal_length": [5.1, 7, 6.3],
    "sepal_width": [3.5, 3.2, 3.3],
    "petal_length": [1.4, 4.7, 6],
    "petal_width": [0.2, 1.4, 2.5],
    "variety": [0, 1, 2]
}


class TestPostgreSQL2Tfrecord:

    def test_format_data(self):
        res = sql2tfrecord.format_data(test_data)
        assert res == test_format_data

    def test_write2tfrecord(self):
        save_data_dir = os.path.join(package_dir, "test_result.tfrecord")
        sql2tfrecord.write2tfrecord(test_format_data, save_data_dir)
        assert "test_result.tfrecord" in os.listdir(package_dir)
        os.remove(save_data_dir)
