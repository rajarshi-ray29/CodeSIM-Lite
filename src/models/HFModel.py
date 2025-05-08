import os
import time
import torch
import gc
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .Base import BaseModel

usage_log_file_path = "usage_log.csv"
MODEL_NAME = 'google/gemma-2-9b-it'  # Using the instruction-tuned version

class HFModel(BaseModel):
    def __init__(self, model_name=None, sleep_time=0, quantization="4bit", **kwargs):
        self.model_name = MODEL_NAME
        self.sleep_time = sleep_time
        self.quantization = quantization

        self.temperature = kwargs.get("temperature", 0.0)
        self.top_p = kwargs.get("top_p", 0.95)
        self.top_k = kwargs.get("top_k", 64)
        self.max_tokens = kwargs.get("max_tokens", 4096)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Configure quantization
        quant_config = None
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        
        # Load model with quantization if specified
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map={"": "cuda"}, 
            quantization_config=quant_config if quantization == "4bit" else None,
            torch_dtype=torch.bfloat16 if quantization != "4bit" else None,
            attn_implementation="sdpa"
        )

    def __del__(self):
        # Cleanup when the object is deleted
        self.clear_gpu_memory()

    def clear_gpu_memory(self):
        """Clear model from GPU memory"""
        if hasattr(self, 'model'):
            if self.quantization != "4bit":  # Only move to CPU if not quantized
                self.model = self.model.to('cpu')
            del self.model
            self.model = None
        
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            self.tokenizer = None
            
        # Force garbage collection
        gc.collect()
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def prompt(
        self,
        processed_input: list[dict],
        frequency_penalty=0,
        presence_penalty=0,  # Keep parameter for compatibility but don't use it
    ):
        time.sleep(self.sleep_time)

        start_time = time.perf_counter()

        prompt_str = processed_input[0]['content']
        inputs = self.tokenizer(prompt_str, return_tensors="pt").to("cuda")

        # Count prompt tokens
        prompt_tokens = inputs.input_ids.shape[-1]
        
        # Generate response with KV cache optimization
        with torch.no_grad():
            
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask.to("cuda"),
                max_new_tokens=4096,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=1.0 + frequency_penalty,
                # presence_penalty parameter removed as it's not supported
                use_cache=True,  # Explicitly enable KV cache
                cache_implementation=None
            )    
        # Decode the output
        output_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        end_time = time.perf_counter()
        
        # Count completion tokens
        completion_tokens = len(outputs[0]) - prompt_tokens
        
        # Log usage
        with open(usage_log_file_path, mode="a") as file:
            file.write(f'{self.model_name},{prompt_tokens},{completion_tokens}\n')
        
        run_details = {
            "api_calls": 1,
            "taken_time": end_time - start_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "details": [
                {
                    "model_name": self.model_name,
                    "model_prompt": processed_input,
                    "model_response": output_text,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                }
            ],
        }
        
        return output_text, run_details
