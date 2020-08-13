import tensorflow as tf
from ray import tune
test_data = {
    "sepal_length": [
        5.1, 7, 6.3, 5.1, 7, 6.3, 5.1, 7, 6.3, 5.1, 7, 6.3],
    "sepal_width": [
        3.5, 3.2, 3.3, 3.5, 3.2, 3.3, 3.5, 3.2, 3.3, 3.5, 3.2, 3.3],
    "petal_length": [
        1.4, 4.7, 6, 1.4, 4.7, 6, 1.4, 4.7, 6, 1.4, 4.7, 6],
    "petal_width": [
        0.2, 1.4, 2.5, 0.2, 1.4, 2.5, 0.2, 1.4, 2.5, 0.2, 1.4, 2.5],
    "variety": [
        "Setosa",
        "Versicolor",
        "Virginica",
        "Setosa",
        "Versicolor",
        "Virginica",
        "Setosa",
        "Versicolor",
        "Virginica",
        "Setosa",
        "Versicolor",
        "Virginica"]
}
test_format_data = {
    "sepal_length": [
        5.1, 7, 6.3, 5.1, 7, 6.3, 5.1, 7, 6.3, 5.1, 7, 6.3],
    "sepal_width": [
        3.5, 3.2, 3.3, 3.5, 3.2, 3.3, 3.5, 3.2, 3.3, 3.5, 3.2, 3.3],
    "petal_length": [
        1.4, 4.7, 6, 1.4, 4.7, 6, 1.4, 4.7, 6, 1.4, 4.7, 6],
    "petal_width":  [
        0.2, 1.4, 2.5, 0.2, 1.4, 2.5, 0.2, 1.4, 2.5, 0.2, 1.4, 2.5],
    "variety": tf.keras.utils.to_categorical([
        0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
}

# test_hyperparameter_space = {
#     "lr": tune.sample_from([0.05, 0.01]),
#     "dense_1": tune.sample_from([1, 2]),
#     "dense_2": tune.sample_from([1, 2]),
#     "epochs": tune.sample_from([2, 3]),
#     "batch_size": tune.sample_from([5, 6])
# }

test_hyperparameter_space = {
    "lr": tune.loguniform(0.001, 0.1),
    "dense_1": tune.uniform(2, 128),
    "dense_2": tune.uniform(2, 128),
    "epochs": tune.uniform(1, 5),
    "batch_size": tune.sample_from([16, 32])
}
test_simple_train_hyperparameters = {
    "lr": 0.05,
    "dense_1": 1,
    "dense_2": 1,
    "batch_size": 10,
    "epochs": 1
}
