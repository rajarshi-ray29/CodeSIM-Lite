from models.Gemini import Gemini
from models.OpenAI import OpenAIModel
from models.OpenAI import ChatGPT
from models.OpenAI import GPT4
from models.OpenAI import GPT4T
from models.OpenAI import GPT4o


class ModelFactory:
    @staticmethod
    def get_model_class(model_name: str):
        model_name = model_name.lower()
        if model_name == "gemini":
            return Gemini
        elif model_name == "chatgpt":
            return ChatGPT
        elif model_name == "gpt4":
            return GPT4
        elif model_name == "gpt4t":
            return GPT4T
        elif model_name == "gpt4o":
            return GPT4o
        elif model_name == "openai":
            return OpenAIModel
        else:
            raise Exception(f"Unknown model name {model_name}")
