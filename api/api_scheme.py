from pydantic import BaseModel


class HealthCheckOutput(BaseModel):
    health: bool


class MetricsOutput(BaseModel):
    name: str
    loss: float
    accuracy: float

class RetrainModelOutput(BaseModel):
    train: bool
