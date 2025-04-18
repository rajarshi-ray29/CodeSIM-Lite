import os
import requests
import base64
import json

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .Base import BaseModel

import os
from openai import OpenAI, AzureOpenAI
import time

usage_log_file_path = "anthropic_usage_log.csv"


class AnthropicModel(BaseModel):
    def __init__(
        self,
        model_name,
        sleep_time=0,
        **kwargs
    ):
        if model_name is None:
            raise Exception("Model name is required")

        self.model_name = f"anthropic/{model_name}"

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

        self.temperature = kwargs.get("temperature", 0.0)
        self.top_p = kwargs.get("top_p", 0.95)
        self.max_tokens = kwargs.get("max_tokens", 8000)

        self.sleep_time = sleep_time

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def prompt(
        self,
        processed_input: list[dict],
        frequency_penalty=0,
        presence_penalty=0
    ):
        time.sleep(self.sleep_time)

        start_time = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": processed_input[0]["content"]
                        },
                    ]
                }
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        print(response.choices[0].message.content)

        end_time = time.perf_counter()

        with open(usage_log_file_path, mode="a") as file:
            file.write(f'{self.model_name},{response.usage.prompt_tokens},{response.usage.completion_tokens}\n')
        
        run_details = {
            "api_calls": 1,
            "taken_time": end_time - start_time,

            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,

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
