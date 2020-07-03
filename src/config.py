import os
from ray import tune

predict_categories = ["Setosa", "Versicolor", "Virginica"]
hyperparameter_space = {
    "lr": tune.loguniform(0.001, 0.1),
    "dense_1": tune.uniform(2, 128),
    "dense_2": tune.uniform(2, 128),
    "epochs": tune.uniform(1, 5),
    "batch_size": tune.sample_from([16,32])
}
feature_names = [
    'petal_length',
    'petal_width',
    'sepal_length',
    'sepal_width']
label_name = "variety"
num_samples = 20
