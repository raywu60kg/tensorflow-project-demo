import tensorflow as tf
import ray
from src.model import create_model
from src.config import hyperparameter_space, num_samples


class TrainModel:
    def tuning(self):
        raise NotImplementedError

    def get_best_model(self):
        raise NotImplementedError

    def write2serving_model(self):
        raise NotImplementedError


class TrainKerasModel(TrainModel):
    def tuning(self, hp):

        model = create_model(
            learning_rate=hp["lr"], dense_1=hp["dense_1"], dense_2=hp["dense_2"])
        checkpoint_callback = ModelCheckpoint(
            "model.h5", monitor='loss', save_best_only=True, save_freq=2)

        # Enable Tune to make intermediate decisions by using a Tune
        # Callback hook. This is Keras specific.
        callbacks = [checkpoint_callback, TuneReporterCallback()]

        # Train the model
        model.fit(
            train_x, train_y,
            validation_data=(test_x, test_y),
            verbose=0,
            batch_size=10,
            epochs=20,
            callbacks=callbacks)

    def get_best_model(self):
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
