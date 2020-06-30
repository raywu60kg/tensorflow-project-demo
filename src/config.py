import os
import ray

predict_categories = ["Setosa", "Versicolor", "Virginica"]
hyperparameter_space = {
    "lr": ray.tune.loguniform(0.001, 0.1),
    "dense_1": ray.tune.uniform(2, 128),
    "dense_2": ray.tune.uniform(2, 128),
}
num_samples = 20
