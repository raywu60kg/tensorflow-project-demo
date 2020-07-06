from src.model import KerasModel
from src.pipeline import Pipeline
from src.train import TrainKerasModel
from tests.resources.test_data import test_hyperparameter_space
import ray
import os
package_dir = os.path.dirname(os.path.abspath(__file__))
pipeline = Pipeline(
    tfrecords_filenames=os.path.join(
        package_dir, "resources", "test_data.tfrecord"))
train_keras_model = TrainKerasModel(pipeline=pipeline)


class TestTrainKerasModel:

    # def test_simple_train(self):
    #     model = train_keras_model.simple_train(
    #         {
    #             "lr": 0.05,
    #             "dense_1": 1,
    #             "dense_2": 1,
    #             "batch_size": 5,
    #             "epochs": 2
    #         })
    #     assert model is not None

    def test_get_best_model(self):
        # https://github.com/ray-project/ray/issues/8047
        test_hyperparameter_space.update({"tfrecords_filenames": os.path.join(
            package_dir, "resources", "test_data.tfrecord")})
        print(test_hyperparameter_space)
        tuned_model = train_keras_model.get_best_model(
            hyperparameter_space=test_hyperparameter_space,
            num_samples=1)
        print(tuned_model)
        assert 1 == 2
