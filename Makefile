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

init-training: init-sql run-app
	
