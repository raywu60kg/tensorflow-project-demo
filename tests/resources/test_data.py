import tensorflow as tf
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
