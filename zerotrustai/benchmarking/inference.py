import time
import logging
import torch
from zerotrustai.utils.data import InferenceMetrics

logger = logging.getLogger(__name__)


class InferenceBenchmark:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_text(self, prompt: str, max_tokens: int = 50) -> InferenceMetrics:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        input_length = inputs.input_ids.shape[1]
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                ),
            )

        total_time = time.time() - start_time
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt) :]
        new_tokens = outputs.shape[1] - input_length
        tokens_per_second = new_tokens / total_time if total_time > 0 else 0

        return InferenceMetrics(
            prompt=prompt,
            response=response,
            time_to_first_token=total_time,
            tokens_per_second=tokens_per_second,
            total_time=total_time,
            total_tokens=new_tokens,
        )

    def benchmark_prompts(self, prompts: list[str]) -> list[InferenceMetrics]:
        results = []
        for prompt in prompts:
            logger.info(f"Benchmarking: {prompt[:50]}...")
            metrics = self.generate_text(prompt)
            results.append(metrics)
            logger.info(
                f"Generated {metrics.total_tokens} tokens in {metrics.total_time:.2f}s ({metrics.tokens_per_second:.1f} tok/s)"
            )
        return results
