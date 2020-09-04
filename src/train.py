from src.callback import TuneReporterCallback
from src.model import KerasModel
from src.pipeline import Pipeline
import json
import os
import ray
import tensorflow as tf

keras_model = KerasModel()


class TrainModel:

    def get_best_model(self):
        raise NotImplementedError

    def write2serving_model(self):
        raise NotImplementedError


class TrainKerasModel(TrainModel):
    """Train the tensorflow keras model.

    Attributes:
        pipeline: pipeline object.
    """
    
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def simple_train(self, hp):
        """Fit the model with the data from pipeline
        Args:
            hp: hyperparameters in dictionary format.
        """

        train_dataset = self.pipeline.get_train_data(hp["batch_size"])
        val_dataset = self.pipeline.get_val_data(hp["batch_size"])
        model = keras_model.create_model(
            learning_rate=hp["lr"],
            dense_1=hp["dense_1"],
            dense_2=hp["dense_2"])

        model.fit(
            train_dataset,
            validation_data=val_dataset,
            verbose=1,
            epochs=hp["epochs"])
        return model

    def get_best_model(self, hyperparameter_space, num_samples):
        """Fit the model with data and find the best hyperparameter space
        in the search space using ray tune.
        Args:
            hyperparameter_space: hyperparameter in dictionary format
                with ray tune defined value.
            num_samples: number of model that ray tune will train.
        
        Returns:
            The best keras model within all the search space.
        """

        def tuning(hp):
            import tensorflow as tf
            pipeline = Pipeline(tfrecords_filenames=hp["tfrecords_filenames"])
            train_dataset = pipeline.get_train_data(int(hp["batch_size"]))
            val_dataset = pipeline.get_val_data(int(hp["batch_size"]))
            model = keras_model.create_model(
                learning_rate=float(hp["lr"]),
                dense_1=int(hp["dense_1"]),
                dense_2=int(hp["dense_2"]))
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                "model.h5", monitor='loss', save_best_only=True, save_freq=2)
            callbacks = [checkpoint_callback, TuneReporterCallback()]
            model.fit(
                train_dataset, validation_data=val_dataset,
                verbose=1,
                epochs=int(hp["epochs"]),
                callbacks=callbacks)
        ray.shutdown()
        ray.init(log_to_driver=False)
        analysis = ray.tune.run(
            tuning,
            verbose=1,
            config=hyperparameter_space,
            num_samples=num_samples)
        log_dir = analysis.get_best_logdir("keras_info/val_loss", mode="min")
        tuned_model = tf.keras.models.load_model(log_dir + "/model.h5")
        return tuned_model

    def save_model(self, model, filename):
        """Save the model for the serving usage and
        also write the performance metrics.
        """
        test_dataset = self.pipeline.get_test_data(batch_size=1)
        metrics = model.evaluate(test_dataset)

        model.save(filename, save_format='tf')
        res = {
            "loss": float(metrics[0]),
            "accuracy": float(metrics[1])}
        with open(os.path.join(filename, "metrics.json"), "w") as f:
            json.dump(res, f)
        return res
