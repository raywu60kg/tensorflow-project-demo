from api.api_scheme import (
    HealthCheckOutput, MetricsOutput, RetrainModelOutput)
from fastapi import BackgroundTasks
from fastapi import FastAPI
from src.config import num_samples, hyperparameter_space
from src.pipeline import PostgreSQL2Tfrecord, Pipeline
from src.train import TrainKerasModel
import gc
import json
import logging
import os
import uvicorn
import time

tags_metadata = [
    {
        "name": "Model",
        "description": "Operations for the lightgbm model",
    },
    {
        "name": "Default",
        "description": "Basic function for API"
    }
]

app = FastAPI(
    title="Tensorflow Project Demo",
    description="This is machine learning project API",
    version="0.0.1",
    openapi_tags=tags_metadata)
package_dir = os.path.dirname(os.path.abspath(__file__))
# pipeline = Pipeline(
#     tfrecords_filenames=os.path.join(
#         package_dir, "resources", "test_data.tfrecord"))
# train_keras_model = TrainKerasModel(pipeline=pipeline)


@app.get("/health", response_model=HealthCheckOutput, tags=["Default"])
def health_check():
    return {"health": "True"}


@app.get("/model/metrics", response_model=MetricsOutput, tags=["Model"])
def get_model_metrics():
    models_metrics = {}
    models_dir = os.path.join(package_dir, "..", "models")
    for directory in os.listdir(models_dir):
        try:
            with open(
                    os.path.join(
                        models_dir,
                        directory,
                        "metrics.json"), "r") as f:
                model_metrics = json.load(f)
        except Exception as e:
            logging.error("Error in geting model metrics: {}".format(e))
        models_metrics.update({directory: model_metrics})
        return models_metrics


@ app.put("/model", response_model=RetrainModelOutput, tags=["Model"])
async def retrain_model(background_tasks: BackgroundTasks):
    def task_retrain_model():

        try:
            logging.info("Query data from database")
            sql2tfrecord = PostgreSQL2Tfrecord()
            data = sql2tfrecord.query_db()
            formated_data = sql2tfrecord.format_data(data)
            del data
            gc.collect()

            sql2tfrecord.write2tfrecord(
                data=formated_data,
                filename=os.path.join(
                    package_dir, "..", "data", "data.tfrecord"))

        except Exception as e:
            logging.error("Error in process data to tfrecord: {}".format(e))
            return 0
        logging.critical("Writed data to tfrecord")

        try:
            logging.info("Start initializing training process")
            pipeline = Pipeline(
                tfrecords_filenames=os.path.join(
                    package_dir, "..", "data", "data.tfrecord"))
            train_keras_model = TrainKerasModel(pipeline=pipeline)
            hyperparameter_space.update({
                "tfrecords_filenames": os.path.join(
                    package_dir,
                    "..",
                    "data",
                    "data.tfrecord")
            })
        except Exception as e:
            logging.error(
                "Error in initializing training process: {}".format(e))
            return 0
        try:
            logging.info("Start Searching Best Model")
            best_model = train_keras_model.get_best_model(
                hyperparameter_space=hyperparameter_space,
                num_samples=num_samples)
        except Exception as e:
            logging.error("Error in searching best model: {}".format(e))
            return 0

        try:
            logging.info("Start saving model")
            result = train_keras_model.save_model(
                model=best_model,
                filename=os.path.join(
                    package_dir, "..", "models", str(int(time.time()))))
        except Exception as e:
            logging.error("Error in create data pipeline: {}".format(e))

        logging.critical("Retrain Finish. Training result: {}".format(result))
    background_tasks.add_task(task_retrain_model)

    return {"train": "True"}


if __name__ == "__main__":
    uvicorn.run(app=app)
