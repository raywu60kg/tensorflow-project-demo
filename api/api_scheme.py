from pydantic import BaseModel


class HealthCheckOutput(BaseModel):
    health: bool


class MetricsOutput(BaseModel):
    model_name: str
    log_loss: float
    auc: float
    average_precision: float
