import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential
import gc

from .Base import BaseModel

class HFModel(BaseModel):
    def __init__(self, 
                 model_name="google/gemma-2-9b-it", 
                 quantization="4bit", 
                 device=None,
                 sleep_time=0,
                 **kwargs):
        """
        model_name: HuggingFace model ID (default: instruction-tuned Gemma2-9B)
        quantization: '4bit', '8bit', or None
        device: 'cuda', 'cpu', or None (auto-detect)
        sleep_time: seconds to sleep before each call (for rate limiting)
        """
        self.model_name = model_name
        self.quantization = quantization
        self.sleep_time = sleep_time
        self.device = "cuda"

        # Quantization config
        quant_config = None
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        # Clear any existing GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        # Load tokenizer and model
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quant_config
        )
        self.model.eval()

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def prompt(self, processed_input, max_new_tokens=512, temperature=0.1, top_p=0.95, **kwargs):
        """
        processed_input: list of dicts with 'content' key, e.g. [{"content": "..."}]
        Returns: (generated_text, run_details)
        """
        time.sleep(self.sleep_time)
        start_time = time.perf_counter()

        # Create CodeSIM-style prompt
        codesim_prompt = processed_input[0]['content']
        
        # Tokenize the input
        inputs = self.tokenizer(codesim_prompt, return_tensors="pt").to(self.device)
        
        # Generate with controlled parameters
        with torch.inference_mode():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask.to(self.device),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Get full response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part (remove the input prompt)
        generated_text = full_response[len(codesim_prompt):]

        end_time = time.perf_counter()
        run_details = {
            "api_calls": 1,
            "taken_time": end_time - start_time,
            "prompt_tokens": inputs.input_ids.shape[-1],
            "completion_tokens": outputs.shape[-1] - inputs.input_ids.shape[-1],
            "cost": 0,
            "details": [
                {
                    "model_name": self.model_name,
                    "model_prompt": processed_input,
                    "model_response": generated_text,
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            ],
        }

        # Clean up memory
        del outputs
        del inputs
        torch.cuda.empty_cache()
        gc.collect()
        
        return generated_text, run_details
    
    def __del__(self):
        # Clean up when the instance is deleted
        if hasattr(self, 'model'):
            del self.model
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Example usage
    hfmodel = HFModel()
    processed_input = [{"content": "Write a Python function to check if any two numbers in a list are closer than a given threshold."}]
    response, run_details = hfmodel.prompt(processed_input)
    print(response)
    print(run_details)