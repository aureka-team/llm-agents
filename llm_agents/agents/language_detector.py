from pydantic import BaseModel, StrictStr
from pydantic_extra_types.language_code import LanguageAlpha2

from common.cache import RedisCache
from llm_agents.conf import experts
from llm_agents.meta.interfaces import LLMAgent


class LanguageDetectorInput(BaseModel):
    text: StrictStr


class LanguageDetectorOutput(BaseModel):
    language: LanguageAlpha2


class LanguageDetector(LLMAgent[LanguageDetectorInput, LanguageDetectorOutput]):
    def __init__(
        self,
        conf_path=f"{experts.__path__[0]}/language-detector.yaml",
        max_concurrency: int = 10,
        cache: RedisCache | None = None,
    ):
        super().__init__(
            conf_path=conf_path,
            agent_input=LanguageDetectorInput,
            agent_output=LanguageDetectorOutput,
            max_concurrency=max_concurrency,
            cache=cache,
        )
