import os

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


def get_ionos_model(model_name: str) -> OpenAIModel:
    provider = OpenAIProvider(
        api_key=os.environ["IONOS_TOKEN"],
        base_url="https://openai.inference.de-txl.ionos.com/v1",
    )

    return OpenAIModel(
        model_name=model_name,
        provider=provider,
    )
