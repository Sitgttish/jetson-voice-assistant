from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LLMBase(ABC):
    @abstractmethod
    def generate(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        pass


class HuggingFaceLLM(LLMBase):
    def __init__(self, model_id: str, load_in_4bit: bool = True,
                 max_new_tokens: int = 512, temperature: float = 0.7):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        logger.info(f"Loading model {model_id} (4bit={load_in_4bit})...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quant_config,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        logger.info("Model loaded.")

    def generate(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        import torch

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def create_llm() -> LLMBase:
    import config
    return HuggingFaceLLM(
        model_id=config.MODEL_ID,
        load_in_4bit=config.LOAD_IN_4BIT,
        max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=config.TEMPERATURE,
    )
