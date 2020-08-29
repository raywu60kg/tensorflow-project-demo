# API example
## Health Check
```
curl -X GET "http://localhost:8000/health" -H  "accept: application/json"
```

## Get Model Metrics
```
curl -X GET "http://localhost:8000/model/metrics" -H  "accept: application/json"
```

## Retrain Model
```
curl -X PUT "http://localhost:8000/model" -H  "accept: application/json"
```

## Check Prediction model
```
curl -X GET "http://localhost:8501/v1/models/tensorflow-project-demo" -H  "accept: application/json"
```
## Check Model Features
```
curl -X GET "http://localhost:8501/v1/models/tensorflow-project-demo/metadata" -H  "accept: application/json"
```

## Make prediction
```
curl -X POST "http://localhost:8501/v1/models/models:predict" -d '{"inputs":{"sepal_length":[[0]],"sepal_width":[[0]],"petal_width":[[0]],"petal_length":[[0]]}}' -H  "accept: application/json"
```

Output format
```
{
    "outputs": [
        [
            0.607119262,
            0.392192692,
            0.000688050292
        ]
    ]
}
```