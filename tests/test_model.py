from src.model import KerasModel
import numpy as np
import tensorflow as tf

keras_model = KerasModel()


class TestKerasModel:
    def test_create_model(self):
        model = keras_model.create_model(
            learning_rate=0.05,
            dense_1=1,
            dense_2=1)

        print(model.summary())
        assert round(
            float(
                tf.keras.backend.eval(model.optimizer.lr)),
            3) == 0.05

        assert len(model.layers) == 9
