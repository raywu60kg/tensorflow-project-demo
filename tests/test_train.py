from src.model import KerasModel
from src.pipeline import Pipeline
from src.train import TrainKerasModel
import ray
import os
package_dir = os.path.dirname(os.path.abspath(__file__))
pipeline = Pipeline(
    tfrecords_filenames=os.path.join(
        package_dir, "resources", "test_data.tfrecord"))
train_keras_model = TrainKerasModel(pipeline=pipeline)


class TestTrainKerasModel:

    def test_simple_train(self):
        model = train_keras_model.simple_train(
            {
                "lr": 0.05,
                "dense_1": 10,
                "dense_2": 10,
                "batch_size": 5,
                "epochs": 10
            })
        assert 1 == 2
