from pydantic_ai import NativeOutput
from pydantic_ai.models import Model
from pydantic import BaseModel, Field
from pydantic_extra_types.language_code import LanguageAlpha2

from common.cache import RedisCache
from llm_agents.agents import language_detector
from llm_agents.meta.interfaces import LLMAgent


class LanguageDetectorOutput(BaseModel):
    language: LanguageAlpha2 = Field(
        description="The primary language of the given text."
    )


class LanguageDetector(LLMAgent[None, LanguageDetectorOutput]):
    def __init__(
        self,
        conf_path: str = f"{language_detector.__path__[0]}/language-detector.yaml",
        model: Model | None = None,
        max_concurrency: int = 10,
        cache: RedisCache | None = None,
    ):
        super().__init__(
            conf_path=conf_path,
            output_type=NativeOutput(LanguageDetectorOutput),  # type: ignore
            model=model,
            max_concurrency=max_concurrency,
            cache=cache,
        )
