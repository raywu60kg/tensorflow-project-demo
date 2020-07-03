from src.pipeline import PostgreSQL2Tfrecord
from src.pipeline import Pipeline
from src.config import feature_names, label_name
from tests.resources.test_data import test_data, test_format_data
import os

package_dir = os.path.dirname(os.path.abspath(__file__))

sql2tfrecord = PostgreSQL2Tfrecord()
pipeline = Pipeline(
    tfrecords_filenames=os.path.join(
        package_dir,
        "resources",
        "test_data.tfrecord"))


class TestPostgreSQL2Tfrecord:

    def test_format_data(self):
        res = sql2tfrecord.format_data(test_data)
        for key in res.keys():
            if key is not label_name:
                assert res[key] == test_format_data[key]

    def test_write2tfrecord(self):
        save_data_dir = os.path.join(package_dir, "test_data.tfrecord")
        sql2tfrecord.write2tfrecord(test_format_data, save_data_dir)
        assert "test_data.tfrecord" in os.listdir(package_dir)
        # os.remove(save_data_dir)


class TestPipeline:

    def test_get_train_data(self):
        train_data = pipeline.get_train_data(batch_size=5)
        data_size = 0
        for row in train_data:
            print("###", row)
            data_size += 1
        for feature_name in feature_names:
            assert feature_name in train_data.element_spec[0].keys()
        print("@@@@", pipeline.data_size, pipeline.train_size)
        assert data_size == 2

    def test_get_val_data(self):
        val_data = pipeline.get_val_data(batch_size=5)
        data_size = 0
        for row in val_data:
            data_size += 1
        for feature_name in feature_names:
            assert feature_name in val_data.element_spec[0].keys()
        assert data_size == 1

    def test_get_test_data(self):
        test_data = pipeline.get_test_data(batch_size=5)
        data_size = 0
        for row in test_data:
            data_size += 1
        for feature_name in feature_names:
            assert feature_name in test_data.element_spec[0].keys()
        assert data_size == 1
