# ml-model-template
For this project, I used the classic datasets iris to do the project demo. My personal goal for this project is to not only explore the data and build models, but to also build an API server with retrainable model. To achieve this goal, I used fastapi, tensorflow and ray tune.

Also, I decided to develop this project to be the same as how data-related projects are developed in real-world scenarios, wherein the end goal of development is a project that is feasible for production. Therefore, I have put efforts on creating:

1. Exploratory Data Analysis (EDA) in the `notebooks/` folder;
2. An API Server inside the `api/` folder;
3. Files for deployment such as Dockerfile and docker-compose.yml;
4. Documentations in the `docs/` folder; and
5. Some necessary scripts in `scripts/` folder.

In the tensorflow part, I used very powerful tools:

1. tfrecord
2. tensorboard
3. tensorflow serving

for building the project. 

I have another project use lightgbm as the back bone model. Toke a look about the project [**lightgbm-project-demo**](https://github.com/raywu60kg/lightgbm-project-demo) 

## How to run this demo

I have three services: `db`, `serving` and `training`. The training service is a machine learning API that is open on port 8000. I used fastapi for the API server, so you can check it on **http://localhost:8000/docs** after you run the `training` service. The `db` service is 
### 1. Install requirement
- docker
- docker-compose

## 2. Create folder for prediction models 
```
mkdir /opt/models
chmod 777 /opt/models
```
## 3. Start the services
```
docker-compose up -d
```

## 4. Try to update the model
```
curl -X PUT "http://localhost:8000/model" -H  "accept: application/json"
```

## 5. Check Training Result
#### 5.1 Activate tensorboard
```
docker-compose exec training make activate-tensorboard
```

#### 5.2 Check the Result
Turn on your browser and go to **http://localhost:6006/**

## 6. Let's make some prediction
```
curl
```

## Here are some documentations

## Reference
https://dzone.com/articles/data-science-project-folder-structure

https://github.com/drivendata/cookiecutter-data-science

https://colab.research.google.com/github/ray-project/tutorial/blob/mastertune_exercises/exercise_1_basics.ipynb