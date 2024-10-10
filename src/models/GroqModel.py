from typing import List, Dict
import os
import requests
import base64
import json

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .Base import BaseModel

from .groqapi import *

import os
import time
from groq import Groq


class GroqModel(BaseModel):
    def __init__(self, sleep_time=0, **kwargs):
        print(kwargs)
        self.model_name = kwargs.get("model_name", "llama3-8b-8192")
        self.temperature = kwargs.get("temperature", 0.0)
        self.top_p = kwargs.get("top_p", 0.95)
        self.max_tokens = kwargs.get("max_tokens", 8192)
        self.sleep_time = sleep_time

        self.client = Groq(api_key=self.api_key)

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


class LLaMa370B(GroqModel):
    def __init__(self, **kwargs):
        self.api_key = api_key
        super().__init__(model_name="llama-3.1-70b-versatile", max_tokens=8192, **kwargs)

class LLaMa38B(GroqModel):
    def __init__(self, **kwargs):
        self.api_key = api_key
        super().__init__(model_name="llama-3.1-8b-instant", max_tokens=8192, **kwargs)

class Mixtral87B(GroqModel):
    def __init__(self, **kwargs):
        self.api_key = api_key_2
        super().__init__(model_name="mixtral-8x7b-32768", max_tokens=32768, **kwargs)

class Gemma29B(GroqModel):
    def __init__(self, **kwargs):
        self.api_key = api_key_4
        super().__init__(model_name="gemma2-9b-it", max_tokens=4096, **kwargs)

