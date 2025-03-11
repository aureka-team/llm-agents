from pydantic import BaseModel, StrictStr
from pydantic_extra_types.language_code import LanguageName

from common.cache import RedisCache
from common.logger import get_logger

from llm_agents.conf import experts
from llm_agents.meta.interfaces import LLMAgent


logger = get_logger(__name__)


class LanguageTranslatorInput(BaseModel):
    text: StrictStr
    source_language: LanguageName
    target_language: LanguageName


class LanguageTranslatorOutput(BaseModel):
    translation: StrictStr


class LanguageTranslator(
    LLMAgent[
        LanguageTranslatorInput,
        LanguageTranslatorOutput,
    ]
):
    def __init__(
        self,
        conf_path=f"{experts.__path__[0]}/language-translator.yaml",
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
