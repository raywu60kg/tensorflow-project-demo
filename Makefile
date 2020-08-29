IMAGE_NAME=tensorflow-project-template:0.0.1

build:
	echo ${IMAGE_NAME}
	docker build -f ./docker/dockerfile -t ${IMAGE_NAME} .

activate-tensorboard:
	tensorboard --logdir ~/ray_results/tuning

run-app-dev:
	uvicorn api.app:app --reload

run-app:
	uvicorn api.app:app --host 0.0.0.0 --port 8000

init-sql:
	python -m scripts.csv2sql

deploy-default-model:
	cp -r default_model/* /opt/tensorflow-project-demo/

init-training: init-sql deploy-default-model run-app
	
serving-health-check:
	curl http://localhost:8501/v1/models/tensorflow-project-demo

predict:
	curl -X POST "http://localhost:8501/v1/models/tensorflow-project-demo:predict" -d '{"inputs":{"sepal_length":[[0]],"sepal_width":[[0]],"petal_width":[[0]],"petal_length":[[0]]}}' -H  "accept: application/json"