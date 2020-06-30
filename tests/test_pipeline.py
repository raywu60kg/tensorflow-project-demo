from src.pipeline import PostgreSQL2Tfrecord
from src.pipeline import Pipeline
from src.config import epochs
import os

package_dir = os.path.dirname(os.path.abspath(__file__))

sql2tfrecord = PostgreSQL2Tfrecord()
pipeline = Pipeline(
    filenames=os.path.join(package_dir, "resources", "test_data.tfrecord"),
    epochs=5)
test_data = {
    "sepal_length": [5.1, 7, 6.3, 5.1, 7, 6.3, 5.1, 7, 6.3, 5.1, 7, 6.3],
    "sepal_width": [3.5, 3.2, 3.3, 3.5, 3.2, 3.3, 3.5, 3.2, 3.3, 3.5, 3.2, 3.3],
    "petal_length": [1.4, 4.7, 6, 1.4, 4.7, 6, 1.4, 4.7, 6, 1.4, 4.7, 6],
    "petal_width": [0.2, 1.4, 2.5, 0.2, 1.4, 2.5, 0.2, 1.4, 2.5, 0.2, 1.4, 2.5],
    "variety": ["Setosa", "Versicolor", "Virginica", "Setosa", "Versicolor", "Virginica", "Setosa", "Versicolor", "Virginica", "Setosa", "Versicolor", "Virginica"]
}
test_format_data = {
    "sepal_length": [5.1, 7, 6.3, 5.1, 7, 6.3, 5.1, 7, 6.3, 5.1, 7, 6.3],
    "sepal_width": [3.5, 3.2, 3.3, 3.5, 3.2, 3.3, 3.5, 3.2, 3.3, 3.5, 3.2, 3.3],
    "petal_length": [1.4, 4.7, 6, 1.4, 4.7, 6, 1.4, 4.7, 6, 1.4, 4.7, 6],
    "petal_width":  [0.2, 1.4, 2.5, 0.2, 1.4, 2.5, 0.2, 1.4, 2.5, 0.2, 1.4, 2.5],
    "variety": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
}


class TestPostgreSQL2Tfrecord:

    def test_format_data(self):
        res = sql2tfrecord.format_data(test_data)
        assert res == test_format_data

    def test_write2tfrecord(self):
        save_data_dir = os.path.join(package_dir, "test_data.tfrecord")
        sql2tfrecord.write2tfrecord(test_format_data, save_data_dir)
        assert "test_data.tfrecord" in os.listdir(package_dir)
        os.remove(save_data_dir)


class TestPipeline:

    def test_get_train_data(self):
        train_data = pipeline.get_train_data()
        data_size = 0
        for row in train_data:
            data_size += 1
        assert data_size == 41

    def test_get_val_data(self):
        val_data = pipeline.get_val_data()
        data_size = 0
        for row in val_data:
            data_size += 1
        assert data_size == 10

    def test_get_test_data(self):
        test_data = pipeline.get_test_data()
        data_size = 0
        for row in test_data:
            data_size += 1
        assert data_size == 9
