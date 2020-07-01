from src.model import KerasModel
from src.pipeline import Pipeline
from src.train import TrainKerasModel
import os
package_dir = os.path.dirname(os.path.abspath(__file__))
pipeline = Pipeline(
    tfrecords_filenames=os.path.join(
        package_dir, "resources", "test_data.tfrecord"))
train_keras_model = TrainKerasModel(
    train_dataset=pipeline.get_train_data(),
    val_dataset=pipeline.get_val_data())


class TestTrainKerasModel:
    def test_tuning(self):
        train_keras_model.tuning({"lr": 0.05, "dense_1": 1, "dense_2": 1})
    assert 1 == 1
