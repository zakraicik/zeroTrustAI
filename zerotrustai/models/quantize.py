import time
import logging
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False

from zerotrustai.models.config import TARGET_MODELS
from zerotrustai.utils.data import ModelMetrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QuantizedModelLoader:
    def __init__(self, cache_dir: str = "./data/models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: dict[str, any] = {}
        self.model_metrics: dict[str, ModelMetrics] = {}

    def load_fp16_model(self, model_name: str) -> tuple[any, any, ModelMetrics]:
        """Load model with FP16 quantization."""
        if not QUANTIZATION_AVAILABLE:
            raise RuntimeError("transformers not available")

        model_config = TARGET_MODELS.get(model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")

        logger.info(f"Loading {model_config.name} with FP16 quantization...")
        start_time = time.time()

        # Load with FP16
        model = AutoModelForCausalLM.from_pretrained(
            model_config.huggingface_id, torch_dtype=torch.float16, device_map="cpu"
        )

        tokenizer = AutoTokenizer.from_pretrained(model_config.huggingface_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_time = time.time() - start_time

        # Calculate metrics
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = (param_count * 2) / (1024 * 1024)  # FP16 = 2 bytes per param

        metrics = ModelMetrics(
            load_time_seconds=load_time,
            model_size_mb=model_size_mb,
            parameter_count=param_count,
        )

        # Cache
        cache_key = f"{model_name}_fp16"
        self.loaded_models[cache_key] = {"model": model, "tokenizer": tokenizer}
        self.model_metrics[cache_key] = metrics

        logger.info(
            f"✅ FP16 Model: {model_size_mb:.1f}MB, {param_count:,} params, {load_time:.2f}s"
        )
        return model, tokenizer, metrics

    def save_quantized_model(self, model, tokenizer, model_name: str):
        """Save quantized model locally."""
        save_path = self.cache_dir / f"{model_name}_fp16"
        save_path.mkdir(exist_ok=True)

        logger.info(f"Saving quantized model to {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        return str(save_path)

    def quantize_all_models(self):
        """Quantize all target models to FP16."""
        results = {}

        for model_name in TARGET_MODELS.keys():
            try:
                logger.info(f"\n🔄 Quantizing {model_name}...")
                model, tokenizer, metrics = self.load_fp16_model(model_name)

                # Save quantized version
                save_path = self.save_quantized_model(model, tokenizer, model_name)

                results[model_name] = {
                    "status": "success",
                    "metrics": metrics,
                    "save_path": save_path,
                    "compression_ratio": TARGET_MODELS[model_name].expected_size_mb
                    / metrics.model_size_mb,
                }

                # Cleanup
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            except Exception as e:
                logger.error(f"Failed to quantize {model_name}: {e}")
                results[model_name] = {"status": "error", "error": str(e)}

        return results
