import tensorflow as tf
import ray
from src.model import KerasModel
from src.config import hyperparameter_space, num_samples
from src.pipeline import Pipeline
from src.call_back import TuneReporterCallback

keras_model = KerasModel()


class TrainModel:

    def get_best_model(self):
        raise NotImplementedError

    def write2serving_model(self):
        raise NotImplementedError


class TrainKerasModel(TrainModel):
    def __init__(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def simple_train(self, hp):
        model = keras_model.create_model(
            learning_rate=hp["lr"],
            dense_1=hp["dense_1"],
            dense_2=hp["dense_2"])
        model.fit(self.train_dataset)

    def tuning(self, hp):
        model = keras_model.create_model(
            learning_rate=hp["lr"],
            dense_1=hp["dense_1"],
            dense_2=hp["dense_2"])
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            "model.h5", monitor='loss', save_best_only=True, save_freq=2)

        # Enable Tune to make intermediate decisions by using a Tune
        # Callback hook. This is Keras specific.
        callbacks = [checkpoint_callback, TuneReporterCallback()]

        # Train the model
        model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            verbose=1,
            epochs=10,
            callbacks=callbacks)
        print("@@@", self.train_dataset.element_spec)
        # model.fit(
        #     self.train_dataset,
        #     verbose=1,
        #     epochs=10)

    def get_best_model(self, hyperparameter_space, num_samples):
        ray.shutdown()
        ray.init(log_to_driver=False)
        analysis = ray.tune.run(
            self.tuning, verbose=1,
            config=hyperparameter_space,
            num_samples=num_samples)
        log_dir = analysis.get_best_logdir("keras_info/val_loss", mode="min")
        tuned_model = tf.keras.models.load_model(logdir + "/model.h5")
        return tuned_model

    def write2serving_model(self, model, filename):
        model.save(filename, save_format='tf')
