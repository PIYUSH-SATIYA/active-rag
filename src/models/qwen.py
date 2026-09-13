from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class GeneratorResult:
    text: str
    token_ids: list[int]
    tokens: list[str]
    probabilities: list[float]
    offsets: Optional[list[tuple[int, int]]] = None
    finish_reason: Optional[str] = None


class QwenModel:
    """
    Backend for Qwen3-4B-Instruct-2507.

    The tokenizer and model can be injected for CPU/mock testing.
    Use from_pretrained() to load the real Hugging Face model.
    """

    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

    def __init__(self, tokenizer: Any, model: Any):
        self.tokenizer = tokenizer
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = MODEL_NAME,
        quantize_4bit: bool = False,
        device_map: str = "auto",
    ) -> "QwenModel":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_kwargs = {
            "device_map": device_map,
        }

        if quantize_4bit:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            model_kwargs["quantization_config"] = quant_config

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )

        return cls(
            tokenizer=tokenizer,
            model=model,
        )

    def generate(self, prompt: str, **kwargs) -> GeneratorResult:
        inputs = self.tokenizer(prompt, return_tensors="pt")

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        outputs = self.model.generate(
            **inputs,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
        )

        generated_ids = outputs.sequences[0]
        input_length = inputs["input_ids"].shape[1]
        generated_ids = generated_ids[input_length:]

        tokens = [
            self.tokenizer.decode([token_id])
            for token_id in generated_ids
        ]

        text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        probabilities = []

        for token_id, scores in zip(generated_ids, outputs.scores):
            probs = scores.softmax(dim=-1)
            probability = probs[0, token_id].item()
            probabilities.append(probability)

        return GeneratorResult(
            text=text,
            token_ids=[token_id.item() for token_id in generated_ids],
            tokens=tokens,
            probabilities=probabilities,
        )
