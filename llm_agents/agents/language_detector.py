from pydantic_ai import ToolOutput
from pydantic import BaseModel, Field
from pydantic_extra_types.language_code import LanguageAlpha2

from common.cache import RedisCache
from llm_agents.conf import agents
from llm_agents.meta.interfaces import LLMAgent


class LanguageDetectorOutput(BaseModel):
    language: LanguageAlpha2 = Field(
        description="The primary language of the given text."
    )


class LanguageDetector(LLMAgent[None, LanguageDetectorOutput]):
    def __init__(
        self,
        conf_path: str = f"{list(agents.__path__)[0]}/language-detector.yaml",
        max_concurrency: int = 10,
        cache: RedisCache | None = None,
    ):
        super().__init__(
            conf_path=conf_path,
            output_type=ToolOutput(LanguageDetectorOutput),
            max_concurrency=max_concurrency,
            cache=cache,
        )
