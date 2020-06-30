import tensorflow as tf


class Model:
    def create_model(self):
        raise NotImplementedError


class KerasModel(Model):
    def create_model(self):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(int(dense_1), input_shape=(
            4,), activation='relu', name='fc1'))
        model.add(tf.keras.layers.Dense(
            int(dense_2), activation='relu', name='fc2'))
        model.add(tf.keras.layers.Dense(
            3, activation='softmax', name='output'))
        optimizer = SGD(lr=learning_rate)
        model.compile(
            optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        return model
