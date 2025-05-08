import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .Base import BaseModel


class HFModel(BaseModel):
    def __init__(self, model_name, sleep_time=0, **kwargs):
        if model_name is None:
            raise ValueError("Model name is required")
        self.model_name = model_name
        self.temperature = kwargs.get("temperature", 0.0)
        self.top_p = kwargs.get("top_p", 0.95)
        # max_tokens maps to max_output_tokens
        self.max_tokens = kwargs.get("max_tokens", None)
        self.sleep_time = sleep_time

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        self.model.eval()

        # Determine device for generation
        self.device = next(self.model.parameters()).device

    def prompt(
        self,
        processed_input: list[dict],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> tuple[str, dict]:
        # Mirror Gemini prompt: pass raw content
        time.sleep(self.sleep_time)
        start_time = time.perf_counter()

        # Directly use the first message content
        prompt_text = processed_input[0].get("content", "")

        # Tokenize and generate
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt"
        ).to(self.device)
        gen_kwargs = {
            "max_new_tokens": self.max_tokens or self.model.config.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode the full output
        gen_tokens = outputs[0]
        decoded = self.tokenizer.decode(
            gen_tokens,
            skip_special_tokens=True
        )

        end_time = time.perf_counter()
        prompt_tokens = inputs["input_ids"].shape[-1]
        completion_tokens = decoded.split().__len__()  # rough token count fallback

        run_details = {
            "api_calls": 1,
            "taken_time": end_time - start_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": 0,
            "details": [
                {
                    "model_name": self.model_name,
                    "model_prompt": processed_input,
                    "model_response": decoded,
                    "max_tokens": gen_kwargs["max_new_tokens"],
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
            ],
        }

        return decoded, run_details
