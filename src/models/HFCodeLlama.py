import time
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from tenacity import retry, stop_after_attempt, wait_random_exponential
from .Base import BaseModel


class HFModel(BaseModel):
    """
    Loads a Hugging-Face causal-LM in full-precision (fp32) or GPU-friendly
    fp16/bf16 and drives it at maximum GPU throughput.

    Args
    ----
    model_name : str   – Hugging Face model repo (e.g. "codellama/CodeLlama-7b-Instruct-hf")
    dtype      : str   – "fp32", "fp16", or "bf16"  (default: "fp16")
    device     : str   – "cuda" | "cuda:0" | "cpu"  (default: first CUDA GPU)
    """

    def __init__(
        self,
        model_name: str,
        sleep_time: float = 0.0,
        dtype: str = "fp16",
        device: str | None = None,
        **kwargs,
    ):
        if not model_name:
            raise ValueError("Model name is required")

        self.model_name = model_name
        self.temperature = kwargs.get("temperature", 0.0)
        self.top_p = kwargs.get("top_p", 0.95)
        self.max_tokens = kwargs.get("max_tokens", 1024)  # sane default
        self.sleep_time = sleep_time

        # ---------- TOKENIZER ----------
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:  # guarantee a pad token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ---------- MODEL ----------
        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[dtype.lower()]

        # Automatically pick the first visible GPU unless overridden
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=None,          # load once, then .to(device) below
            trust_remote_code=True,   # needed for some HF repos
        ).to(target_device)
        self.model.eval()

        # Optional: compile for a tiny extra speed-up on Ampere+
        try:
            self.model = torch.compile(self.model)
        except Exception:
            pass  # torch<2.1 or unsupported GPU

        self.device = target_device
        self.torch_dtype = torch_dtype

    # -------------------------------------------------------------------------
    # GENERATION
    # -------------------------------------------------------------------------
    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def prompt(
        self,
        processed_input: List[Dict],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> tuple[str, dict]:
        if self.sleep_time:
            time.sleep(self.sleep_time)

        start = time.perf_counter()

        prompt_str = "".join(f"{m['role']}: {m['content']}\n" for m in processed_input)

        inputs = self.tokenizer(
            prompt_str,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        # Autocast gives free speed on fp16/bf16 without precision loss in decoding
        autocast_dtype = (
            torch.float16 if self.torch_dtype == torch.float16 else torch.bfloat16
        )
        gen_args = dict(
            max_new_tokens=self.max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=self.torch_dtype != torch.float32, dtype=autocast_dtype
        ):
            output_ids = self.model.generate(**inputs, **gen_args)

        gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        end = time.perf_counter()
        run_details = {
            "api_calls": 0,
            "taken_time": end - start,
            "prompt_tokens": int(inputs["input_ids"].shape[-1]),
            "completion_tokens": int(gen_ids.shape[-1]),
            "cost": 0,
            "details": [
                {
                    "model_name": self.model_name,
                    "model_prompt": processed_input,
                    "model_response": generated_text,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                }
            ],
        }

        return generated_text, run_details
