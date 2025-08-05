import os

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


OLLAMA_DSN = os.getenv("OLLAMA_DSN", "http://llm-agents-ollama:11434/v1")


def get_ionos_model(model_name: str) -> OpenAIModel:
    provider = OpenAIProvider(
        api_key=os.environ["IONOS_TOKEN"],
        base_url="https://openai.inference.de-txl.ionos.com/v1",
    )

    return OpenAIModel(
        model_name=model_name,
        provider=provider,
    )


def get_ollama_model(model_name: str) -> OpenAIModel:
    provider = OpenAIProvider(base_url=OLLAMA_DSN)
    return OpenAIModel(
        model_name=model_name,
        provider=provider,
    )
