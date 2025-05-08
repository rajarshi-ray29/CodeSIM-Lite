import os
import time
import torch
import gc
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoModelForCausalLM, AutoTokenizer

from .Base import BaseModel

usage_log_file_path = "usage_log.csv"
MODEL_NAME = 'google/gemma-2-9b-it' # Using the instruction-tuned version

class HFModel(BaseModel):
def __init__(self, model_name=None, sleep_time=0, **kwargs):
self.model_name = MODEL_NAME
self.sleep_time = sleep_time

self.temperature = kwargs.get("temperature", 0.0)
self.top_p = kwargs.get("top_p", 0.95)
self.max_tokens = kwargs.get("max_tokens", 4096)

# Load model and tokenizer
self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
self.model = AutoModelForCausalLM.from_pretrained(
self.model_name,
torch_dtype=torch.bfloat16,
device_map="cuda", # Explicitly use CUDA
attn_implementation="sdpa"
)

def __del__(self):
# Cleanup when the object is deleted
self.clear_gpu_memory()

def clear_gpu_memory(self):
"""Clear model from GPU memory"""
if hasattr(self, 'model'):
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
presence_penalty=0, # Keep parameter for compatibility but don't use it
):
time.sleep(self.sleep_time)

start_time = time.perf_counter()

# Apply chat template to format messages correctly for Gemma 2
input_ids = self.tokenizer.apply_chat_template(
processed_input,
return_tensors="pt"
).to("cuda") # Explicitly use CUDA

# Count prompt tokens
prompt_tokens = len(input_ids[0])

# Generate response with KV cache optimization
with torch.no_grad():
# Set do_sample based on temperature
do_sample = self.temperature > 0

outputs = self.model.generate(
input_ids,
max_new_tokens=self.max_tokens,
temperature=self.temperature if do_sample else 1.0,
top_p=self.top_p if do_sample else 1.0,
do_sample=do_sample,
repetition_penalty=1.0 + frequency_penalty,
# presence_penalty parameter removed as it's not supported
use_cache=True, # Explicitly enable KV cache
cache_implementation="offloaded" if torch.cuda.is_available() else None
)
# Decode the output
output_text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

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