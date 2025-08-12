from dataclasses import dataclass
from enum import Enum


class DeviceTier(Enum):
    BUDGET = "budget"
    MID_RANGE = "mid_range"
    FLAGSHIP = "flagship"


@dataclass
class ModelConfig:
    name: str
    huggingface_id: str
    parameters: int
    target_devices: list[DeviceTier]
    expected_size_mb: int


TARGET_MODELS: dict[str, ModelConfig] = {
    "smollm_135m": ModelConfig(
        name="SmolLM-135M",
        huggingface_id="HuggingFaceTB/SmolLM-135M-Instruct",
        parameters=135_000_000,
        target_devices=[DeviceTier.BUDGET, DeviceTier.MID_RANGE, DeviceTier.FLAGSHIP],
        expected_size_mb=540,
    ),
    "smollm_360m": ModelConfig(
        name="SmolLM-360M",
        huggingface_id="HuggingFaceTB/SmolLM-360M-Instruct",
        parameters=360_000_000,
        target_devices=[DeviceTier.MID_RANGE, DeviceTier.FLAGSHIP],
        expected_size_mb=1400,
    ),
    "phi3_mini": ModelConfig(
        name="Phi-3-mini",
        huggingface_id="microsoft/Phi-3-mini-4k-instruct",
        parameters=3_800_000_000,
        target_devices=[DeviceTier.FLAGSHIP],
        expected_size_mb=15200,
    ),
    "smollm2_135m": ModelConfig(
        name="SmolLM2-135M",
        huggingface_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        parameters=135_000_000,
        target_devices=[DeviceTier.BUDGET, DeviceTier.MID_RANGE, DeviceTier.FLAGSHIP],
        expected_size_mb=540,
    ),
}
