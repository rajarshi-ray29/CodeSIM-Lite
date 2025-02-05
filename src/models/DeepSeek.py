import os
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential
import os
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

from .Base import BaseModel
from .deepseekapi import *


class DeepSeek(BaseModel):
    def __init__(self, sleep_time=0, **kwargs):
        self.api_key = API_KEY
        self.sleep_time = sleep_time

        if not self.api_key:
            raise Exception("A key should be provided to invoke the endpoint")

        self.client = ChatCompletionsClient(
            endpoint=API_ENDPOINT,
            credential=AzureKeyCredential(self.api_key)
        )
        self.model_info = self.client.get_model_info()
        self.model_name = self.model_info.model_name
        
        self.temperature = kwargs.get("temperature", 0.0)
        self.top_p = kwargs.get("top_p", 0.95)
        self.max_tokens = kwargs.get("max_tokens", 2048)


    # @retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(3))
    def prompt(
        self,
        processed_input,
        frequency_penalty=0,
        presence_penalty=0
    ):
        # print(processed_input, flush=True)

        time.sleep(self.sleep_time)

        start_time = time.perf_counter()

        payload = {
            "messages": processed_input,
            "max_tokens": self.max_tokens
        }
        response = self.client.complete(payload)

        model_output = response.choices[0].message.content
        if '</think>' in model_output:
            model_output = model_output[model_output.find(
            '</think>') + len('</think>'):].strip()

        cost = response.usage.completion_tokens * \
            (2.19/1e6) + response.usage.prompt_tokens * (0.55/1e6)

        end_time = time.perf_counter()

        run_details = {
            "api_calls": 1,
            "taken_time": end_time - start_time,

            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "cost": cost,

            "details": [
                {
                    "model_name": self.model_name,
                    "model_prompt": processed_input,
                    "model_response": model_output,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                }
            ],
        }

        return model_output, run_details


if __name__ == "__main__":
    # Load your API key from the environment variable
    # Create a Gemini instance
    deepSeek = DeepSeek()

    # Sample API call
    processed_input = [
        {
            'role': 'user', 
            'content': 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n\nGenerate Python3 code to solve the above mentioned problem:'
        }
    ]

    response, run_details = deepSeek.prompt(processed_input)

    print(response)
    print(run_details)
