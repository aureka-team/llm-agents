from pydantic_ai import ToolOutput
from pydantic import BaseModel, StrictStr, Field
from pydantic_extra_types.language_code import LanguageName

from common.cache import RedisCache

from llm_agents.conf import agents
from llm_agents.meta.interfaces import LLMAgent


class LanguageTranslatorDeps(BaseModel):
    source_language: LanguageName
    target_language: LanguageName


class LanguageTranslatorOutput(BaseModel):
    translation: StrictStr = Field(description="Translation of the given text.")


class LanguageTranslator(
    LLMAgent[
        LanguageTranslatorDeps,
        LanguageTranslatorOutput,
    ]
):
    def __init__(
        self,
        conf_path: str = f"{list(agents.__path__)[0]}/language-translator.yaml",
        max_concurrency: int = 10,
        cache: RedisCache | None = None,
    ):
        super().__init__(
            conf_path=conf_path,
            deps_type=LanguageTranslatorDeps,
            output_type=ToolOutput(LanguageTranslatorOutput),
            max_concurrency=max_concurrency,
            cache=cache,
        )
