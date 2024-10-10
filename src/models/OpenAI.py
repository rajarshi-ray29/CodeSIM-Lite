import os
import requests
import base64
import json

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .Base import BaseModel

from .openaiapi import *

import os
from openai import AzureOpenAI
import time


class OpenAIModel(BaseModel):
    def __init__(
            self, 
            **kwargs
        ):
        pass

    def prompt(
            self, 
            processed_input: list[dict], 
            frequency_penalty=0, 
            presence_penalty=0
        ):
        pass


class GPTV1Base(OpenAIModel):
    def __init__(self, model, sleep_time=0, **kwargs):
        self.model = model
        self.client = AzureOpenAI(
            azure_endpoint=self.model["end_point"],
            api_version=self.model["api_version"],
            api_key=self.model["api_key"]
        )

        self.temperature = kwargs.get("temperature", 0.0)
        self.top_p = kwargs.get("top_p", 0.95)
        self.max_tokens = kwargs.get("max_tokens", 4096)

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
            model=self.model["deployment"],
            messages=processed_input,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=None,
            stream=False
        )

        end_time = time.perf_counter()

        cost = 0
        cost += (self.model["prompt_token_cost"] *
                 response.usage.prompt_tokens) / 1e6
        cost += (self.model["completion_token_cost"] *
                 response.usage.completion_tokens) / 1e6

        with open(cost_log_file_path, mode="a") as file:
            file.write(f'{self.model["name"]},{response.usage.prompt_tokens},{response.usage.completion_tokens},{cost}\n')
        
        run_details = {
            "api_calls": 1,
            "taken_time": end_time - start_time,

            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "cost": cost,

            "details": [
                {
                    "model_name": self.model["name"],
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


class ChatGPT(GPTV1Base):
    def __init__(self, **kwargs):
        super().__init__(model=chatgpt, sleep_time=0, **kwargs)


class ChatGPT2(GPTV1Base):
    def __init__(self, **kwargs):
        super().__init__(model=chatgpt2, sleep_time=0, **kwargs)



class ChatGPT3(GPTV1Base):
    def __init__(self, **kwargs):
        super().__init__(model=chatgpt3, sleep_time=0, **kwargs)


class ChatGPT1106(GPTV1Base):
    def __init__(self, **kwargs):
        super().__init__(model=chatgpt1106, sleep_time=0, **kwargs)



class GPT4(GPTV1Base):
    
    def __init__(self, **kwargs):
        super().__init__(model=gpt4, sleep_time=0, **kwargs)


class GPT42(GPTV1Base):
    
    def __init__(self, **kwargs):
        super().__init__(model=gpt42, sleep_time=0, **kwargs)


class GPT43(GPTV1Base):
    
    def __init__(self, **kwargs):
        super().__init__(model=gpt43, sleep_time=0, **kwargs)


class GPT44(GPTV1Base):
    
    def __init__(self, **kwargs):
        super().__init__(model=gpt44, sleep_time=0, **kwargs)


class GPTV2Base(OpenAIModel):
    def __init__(self, model, sleep_time=60, **kwargs):
        self.model = model

        self.headers = {
            "Content-Type": "application/json",
            "api-key": kwargs.get("api-key", self.model["api_key"]),
        }
        self.end_point = kwargs.get("end_point", self.model["end_point"])

        self.temperature = kwargs.get("temperature", 0.0)
        self.top_p = kwargs.get("top_p", 0.95)
        self.max_tokens = kwargs.get("max_tokens", 4096)

        self.sleep_time = sleep_time


    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def prompt(
        self, 
        processed_input: list[dict], 
        frequency_penalty=0, 
        presence_penalty=0
    ):

        time.sleep(self.sleep_time)


        # Payload for the request
        payload = {
            "messages": processed_input,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }

        start_time = time.perf_counter()

        response = requests.post(
            self.end_point, headers=self.headers, json=payload)
        # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        response.raise_for_status()

        end_time = time.perf_counter()

        # Handle the response as needed (e.g., print or process)
        response = response.json()

        cost = 0
        cost += (self.model["prompt_token_cost"] *
                 response["usage"]["prompt_tokens"]) / 1e6
        cost += (self.model["completion_token_cost"] *
                 response["usage"]["completion_tokens"]) / 1e6

        with open(cost_log_file_path, mode="a") as file:
            file.write(f'{self.model["name"]},{response["usage"]["prompt_tokens"]},{response["usage"]["completion_tokens"]},{cost}\n')

        run_details = {
            "api_calls": 1,
            "taken_time": end_time - start_time,

            "prompt_tokens": response["usage"]["prompt_tokens"],
            "completion_tokens": response["usage"]["completion_tokens"],
            "cost": cost,

            "details": [
                {
                    "model_name": self.model["name"],
                    "model_prompt": processed_input,
                    "model_response": response["choices"][0]["message"]["content"],                    
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty
                }
            ],
        }

        return response["choices"][0]["message"]["content"], run_details


class GPT4ol4(GPTV2Base):
    def __init__(self, **kwargs):
        super().__init__(model=gpt4o, **kwargs)


class GPT4ol5(GPTV2Base):
    def __init__(self, **kwargs):
        super().__init__(model=gpt4o2, **kwargs)


class GPT4ol6(GPTV2Base):
    def __init__(self, **kwargs):
        super().__init__(model=gpt4o3, **kwargs)


class GPT4ol(GPTV2Base):
    def __init__(self, **kwargs):
        super().__init__(model=gpt4ol, sleep_time=30, **kwargs)


class GPT4ol2(GPTV2Base):
    def __init__(self, **kwargs):
        super().__init__(model=gpt4ol2, sleep_time=30, **kwargs)


class GPT4ol3(GPTV2Base):
    def __init__(self, **kwargs):
        super().__init__(model=gpt4ol3, sleep_time=30, **kwargs)




class GPT4T(GPTV2Base):
    
    def __init__(self, **kwargs):
        super().__init__(model=gpt4T, **kwargs)

