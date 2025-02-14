from typing import List, Dict
import os
import requests
import base64
import json

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .Base import BaseModel

api_key = os.getenv("GROQ_API_KEY")

import os
import time
from groq import Groq


class GroqModel(BaseModel):
    def __init__(self, model_name, sleep_time=0, **kwargs):
        if model_name is None:
            raise Exception("Model name is required")
        
        self.model_name = model_name        
        self.temperature = kwargs.get("temperature", 0.0)
        self.top_p = kwargs.get("top_p", 0.95)
        self.max_tokens = kwargs.get("max_tokens", 16000)
        self.sleep_time = sleep_time
        self.api_key = api_key

        self.client = Groq(api_key=self.api_key)

    @retry(wait=wait_random_exponential(min=600, max=3600), stop=stop_after_attempt(5))
    def prompt(
        self,
        processed_input: List[Dict],
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ) -> tuple[str, dict]:
        
        time.sleep(self.sleep_time)

        start_time = time.perf_counter()

        response = self.client.chat.completions.create(
            messages=processed_input,
            model=self.model_name,
            max_tokens=self.max_tokens
        )

        end_time = time.perf_counter()

        run_details = {
            "api_calls": 1,
            "taken_time": end_time - start_time,

            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "cost": 0,

            "details": [
                {
                    "model_name": self.model_name,
                    "model_prompt": processed_input,
                    "model_response": response.choices[0].message.content,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty
                }
            ],
        }

        return response.choices[0].message.content, run_details

