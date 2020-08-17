from src.pipeline import Pipeline
from src.train import TrainKerasModel
from tests.resources.test_data import (
    test_hyperparameter_space, test_simple_train_hyperparameters)
import os
import tensorflow as tf
import shutil
package_dir = os.path.dirname(os.path.abspath(__file__))
pipeline = Pipeline(
    tfrecords_filenames=os.path.join(
        package_dir, "resources", "test_data.tfrecord"))
train_keras_model = TrainKerasModel(pipeline=pipeline)


class TestTrainKerasModel:

    def test_simple_train(self):
        model = train_keras_model.simple_train(
            test_simple_train_hyperparameters)
        assert model is not None

    def test_get_best_model(self):
        test_hyperparameter_space.update(
            {"tfrecords_filenames": os.path.join(
                package_dir, "resources", "test_data.tfrecord")})
        tuned_model = train_keras_model.get_best_model(
            hyperparameter_space=test_hyperparameter_space,
            num_samples=1)
        print(tuned_model)
        assert tuned_model is not None

    def test_save_model(self):
        model = tf.keras.models.load_model(
            os.path.join(
                package_dir, "resources", "test_model"))
        train_keras_model.save_model(
            model,
            os.path.join(package_dir, "test_saved_model"))
        list_dir = os.listdir(package_dir)
        print(list_dir)
        assert "test_saved_model" in list_dir
        assert model is not None
        shutil.rmtree(os.path.join(package_dir, "test_saved_model"))
