from pydantic import BaseModel, StrictStr, Field
from pydantic_extra_types.language_code import LanguageName

from common.cache import RedisCache

from llm_agents.conf import agents
from llm_agents.meta.interfaces import LLMAgent


class LanguageTranslatorInput(BaseModel):
    text: StrictStr
    source_language: LanguageName
    target_language: LanguageName


class LanguageTranslatorOutput(BaseModel):
    translation: StrictStr = Field(
        description="Accurate and context-aware translation of the given text."
    )


class LanguageTranslator(
    LLMAgent[
        LanguageTranslatorInput,
        LanguageTranslatorOutput,
    ]
):
    def __init__(
        self,
        conf_path=f"{agents.__path__[0]}/language-translator.yaml",
        max_concurrency: int = 10,
        cache: RedisCache = None,
    ):
        super().__init__(
            conf_path=conf_path,
            agent_input=LanguageTranslatorInput,
            agent_output=LanguageTranslatorOutput,
            max_concurrency=max_concurrency,
            cache=cache,
        )
