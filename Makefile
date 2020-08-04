IMAGE_NAME=ml-project-template:example

build:
	echo ${IMAGE_NAME}
	docker build -f ./Dockerfile -t ${IMAGE_NAME} .
	docker push ${IMAGE_NAME}

activate-tensorboard:
	tensorboard --logdir ~/ray_results/tuning
