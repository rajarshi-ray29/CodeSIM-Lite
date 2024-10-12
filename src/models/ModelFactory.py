from models.Gemini import *
from models.OpenAI import *
from models.GroqModel import *


class ModelFactory:
    @staticmethod
    def get_model_class(model_name: str):
        model_name = model_name.lower()
        if model_name == "gemini":
            return Gemini
        if model_name == "llama70b":
            return LLaMa370B
        if model_name == "llama8b":
            return LLaMa38B
        if model_name == "mixtral":
            return Mixtral87B
        if model_name == "gemma":
            return Gemma29B
        elif model_name == "chatgpt":
            return ChatGPT
        elif model_name == "chatgpt2":
            return ChatGPT2
        elif model_name == "chatgpt3":
            return ChatGPT3
        elif model_name == "chatgpt11061":
            return ChatGPT11061
        elif model_name == "chatgpt11062":
            return ChatGPT11062
        elif model_name == "chatgpt11063":
            return ChatGPT11063
        elif model_name == "gpt41":
            return GPT41
        elif model_name == "gpt42":
            return GPT42
        elif model_name == "gpt43":
            return GPT43
        elif model_name == "gpt44":
            return GPT44
        elif model_name == "gpt4t":
            return GPT4T
        elif model_name == "gpt4ol":
            return GPT4ol
        elif model_name == "gpt4ol2":
            return GPT4ol2
        elif model_name == "gpt4ol3":
            return GPT4ol3
        elif model_name == "gpt4ol4":
            return GPT4ol4
        elif model_name == "gpt4ol5":
            return GPT4ol5
        elif model_name == "gpt4ol6":
            return GPT4ol6
        elif model_name == "openai":
            return OpenAIModel
        else:
            raise Exception(f"Unknown model name {model_name}")
