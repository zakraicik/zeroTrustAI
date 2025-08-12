import os
import time
import logging
from pathlib import Path
from typing import Optional
from zerotrustai.models.config import ModelConfig, TARGET_MODELS
from zerotrustai.utils.data import ModelMetrics

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import snapshot_download

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./data/models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: dict[str, any] = {}
        self.model_metrics: dict[str, ModelMetrics] = {}

    def download_model(self, model_config: ModelConfig) -> str:
        if not HF_AVAILABLE:
            raise RuntimeError("transformers not available - run: poetry install")

        model_cache_path = self.cache_dir / model_config.name

        if model_cache_path.exists():
            logger.info(f"Model {model_config.name} already cached")
            return str(model_cache_path)

        logger.info(f"Downloading {model_config.name}...")
        snapshot_download(
            repo_id=model_config.huggingface_id,
            local_dir=str(model_cache_path),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.onnx", "onnx/*"],
        )

        return str(model_cache_path)

    def get_model_size(self, model_path: str) -> float:
        total_size = 0
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
        return total_size / (1024 * 1024)  # MB

    def load_model(self, model_name: str) -> tuple[any, any, ModelMetrics]:
        if not HF_AVAILABLE:
            raise RuntimeError("transformers not available")

        model_config = TARGET_MODELS.get(model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")

        model_path = self.download_model(model_config)

        logger.info(f"Loading {model_config.name}...")
        start_time = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

        load_time = time.time() - start_time
        model_size = self.get_model_size(model_path)
        param_count = sum(p.numel() for p in model.parameters())

        metrics = ModelMetrics(
            load_time_seconds=load_time,
            model_size_mb=model_size,
            parameter_count=param_count,
        )

        self.loaded_models[model_name] = {"model": model, "tokenizer": tokenizer}
        self.model_metrics[model_name] = metrics

        logger.info(
            f"Loaded: {load_time:.2f}s, {model_size:.1f}MB, {param_count:,} params"
        )

        return model, tokenizer, metrics
