from pydantic import BaseModel


class HealthCheckOutput(BaseModel):
    health: bool


# class MetricsOutput(BaseModel):
#     name: str
#     metrics: dict


class RetrainModelOutput(BaseModel):
    train: bool
