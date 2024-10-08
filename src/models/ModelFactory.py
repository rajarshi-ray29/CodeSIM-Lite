from models.Gemini import *
from models.OpenAI import *
from models.GroqModel import *


class ModelFactory:
    @staticmethod
    def get_model_class(model_name: str):
        model_name = model_name.lower()
        if model_name == "gemini":
            return Gemini
        if model_name == "llama70B":
            return LLaMa370B
        if model_name == "llama8B":
            return LLaMa38B
        if model_name == "mixtral":
            return Mixtral87B
        if model_name == "gemma":
            return Gemma29B
        elif model_name == "chatgpt":
            return ChatGPT
        elif model_name == "chatgpt3":
            return ChatGPT3
        elif model_name == "gpt4":
            return GPT4
        elif model_name == "gpt42":
            return GPT42
        elif model_name == "gpt43":
            return GPT43
        elif model_name == "gpt44":
            return GPT44
        elif model_name == "gpt4t":
            return GPT4T
        elif model_name == "gpt4o":
            return GPT4o
        elif model_name == "gpt4o2":
            return GPT4o2
        elif model_name == "gpt4o3":
            return GPT4o3
        elif model_name == "openai":
            return OpenAIModel
        else:
            raise Exception(f"Unknown model name {model_name}")
