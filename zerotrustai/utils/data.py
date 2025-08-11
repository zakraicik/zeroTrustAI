from dataclasses import dataclass


@dataclass
class ModelMetrics:
    load_time_seconds: float
    model_size_mb: float
    parameter_count: int


@dataclass
class InferenceMetrics:
    prompt: str
    response: str
    time_to_first_token: float
    tokens_per_second: float
    total_time: float
    total_tokens: int
