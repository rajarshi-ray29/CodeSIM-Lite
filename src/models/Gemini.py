import pprint
import os
import google.generativeai as genai
import dotenv
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .Base import BaseModel

from .geminiapi import api_key

genai.configure(api_key=api_key)
model_name="gemini-1.5-flash"

class Gemini(BaseModel):
    def __init__(self, sleep_time=0, **kwargs):
        self.model_name = kwargs.get("model_name", model_name)
        self.api_key = kwargs.get("api_key", api_key)
        self.temperature = kwargs.get("temperature", 0.0)
        self.top_p = kwargs.get("top_p", 0.95)
        self.max_tokens = kwargs.get("max_tokens", 8192)
        self.sleep_time = sleep_time

        genai.configure(api_key=api_key)

        # Create the model
        generation_config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": 64,
            "max_output_tokens": self.max_tokens,
            "response_mime_type": "text/plain",
        }

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
        )
    

    @retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(20))
    def prompt(self, processed_input):
        
        time.sleep(self.sleep_time)

        start_time = time.perf_counter()

        response = self.model.generate_content(processed_input[0]['content'])

        end_time = time.perf_counter()

        run_details = {
            "api_calls": 1,
            "taken_time": end_time - start_time,

            "prompt_tokens": response.usage_metadata.prompt_token_count,
            "completion_tokens": response.usage_metadata.candidates_token_count,
            "cost": 0,

            "details": [
                {
                    "model_name": model_name,
                    "model_prompt": processed_input,
                    "model_response": response.candidates[0].content.parts[0].text,                    
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                }
            ],
        }

        return response.text, run_details


if __name__ == "__main__":
    # Load your API key from the environment variable
    # Create a Gemini instance
    gemini = Gemini()

    # Sample API call
    processed_input = [{"content": "Tell me a joke."}]
    response, run_details = gemini.prompt(processed_input)

    print(response)
    print(run_details)

    