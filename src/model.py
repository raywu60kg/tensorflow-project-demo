import tensorflow as tf
from src.config import feature_names, label_name


class Model:
    def create_model(self):
        raise NotImplementedError


class KerasModel(Model):
    def create_model(self, learning_rate, dense_1, dense_2):

        # model = tf.keras.models.Sequential()
        inputs = {}
        concatenated_feature = []
        for feature_name in feature_names:

            inputs[feature_name] = tf.keras.Input(
                shape=(1), name=feature_name)
            concatenated_feature.append(inputs[feature_name])

        x = tf.keras.layers.concatenate(concatenated_feature, axis=1)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(
            int(dense_1), input_shape=(4,), activation='relu', name='fc1')(x)
        x = tf.keras.layers.Dense(
            int(dense_2), activation='relu', name='fc2')(x)
        x = tf.keras.layers.Dense(3, activation='softmax', name=label_name)(x)
        model = tf.keras.Model(inputs, {label_name: x})

        # model.add(tf.keras.layers.Dense(int(dense_1), input_shape=(
        #     4,), activation='relu', name='fc1'))
        # model.add(tf.keras.layers.Dense(
        #     int(dense_2), activation='relu', name='fc2'))
        # model.add(tf.keras.layers.Dense(
        #     3, activation='softmax', name='output'))
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        model.compile(
            optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        return model
