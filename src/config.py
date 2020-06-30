import os
from ray import tune

predict_categories = ["Setosa", "Versicolor", "Virginica"]
hyperparameter_space = {
    "lr": tune.loguniform(0.001, 0.1),
    "dense_1": tune.uniform(2, 128),
    "dense_2": tune.uniform(2, 128),
}
num_samples = 20
epochs = 5
